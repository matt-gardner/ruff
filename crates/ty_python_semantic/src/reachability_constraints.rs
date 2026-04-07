//! # Reachability constraints
//!
//! See [`ty_semantic_index::reachability_constraints`] for documentation.

use crate::Db;
use crate::dunder_all::dunder_all_names;
use crate::place::{RequiresExplicitReExport, imported_symbol};
use crate::types::narrow::accumulate_constraint;
use crate::types::{
    CallableTypes, IntersectionBuilder, KnownClass, NarrowingConstraint, Type, TypeContext,
    UnionBuilder, UnionType, infer_expression_type, infer_narrowing_constraint,
};
use ruff_text_size::TextRange;
use ty_semantic_index::place::ScopedPlaceId;
use ty_semantic_index::predicate::{
    CallableAndCallExpr, PatternPredicate, PatternPredicateKind, Predicate, PredicateNode,
    Predicates,
};
use ty_semantic_index::reachability_constraints::{
    ALWAYS_FALSE, ALWAYS_TRUE, AMBIGUOUS, ReachabilityConstraints, ScopedReachabilityConstraintId,
};
use ty_semantic_index::{
    BindingWithConstraints, FileScopeId, SemanticIndex, Truthiness, UseDefMap, place_table,
};

fn singleton_to_type(db: &dyn Db, singleton: ruff_python_ast::Singleton) -> Type<'_> {
    let ty = match singleton {
        ruff_python_ast::Singleton::None => Type::none(db),
        ruff_python_ast::Singleton::True => Type::bool_literal(true),
        ruff_python_ast::Singleton::False => Type::bool_literal(false),
    };
    debug_assert!(ty.is_singleton(db));
    ty
}

fn mapping_pattern_type(db: &dyn Db) -> Type<'_> {
    KnownClass::Mapping.to_instance(db).top_materialization(db)
}

/// Turn a `match` pattern kind into a type that represents the set of all values that would definitely
/// match that pattern.
fn pattern_kind_to_type<'db>(db: &'db dyn Db, kind: &PatternPredicateKind<'db>) -> Type<'db> {
    match kind {
        PatternPredicateKind::Singleton(singleton) => singleton_to_type(db, *singleton),
        PatternPredicateKind::Value(value) => {
            let ty = infer_expression_type(db, *value, TypeContext::default());
            // Only return the type if it's single-valued. For non-single-valued types
            // (like `str`), we can't definitively exclude any specific type from
            // subsequent patterns because the pattern could match any value of that type.
            if ty.is_single_valued(db) {
                ty
            } else {
                Type::Never
            }
        }
        PatternPredicateKind::Class(class_expr, kind) => {
            if kind.is_irrefutable() {
                infer_expression_type(db, *class_expr, TypeContext::default())
                    .to_instance(db)
                    .unwrap_or(Type::Never)
                    .top_materialization(db)
            } else {
                Type::Never
            }
        }
        PatternPredicateKind::Mapping(kind) => {
            if kind.is_irrefutable() {
                mapping_pattern_type(db)
            } else {
                Type::Never
            }
        }
        PatternPredicateKind::Or(predicates) => {
            UnionType::from_elements(db, predicates.iter().map(|p| pattern_kind_to_type(db, p)))
        }
        PatternPredicateKind::As(pattern, _) => pattern
            .as_deref()
            .map(|p| pattern_kind_to_type(db, p))
            .unwrap_or_else(Type::object),
        PatternPredicateKind::Unsupported => Type::Never,
    }
}

/// Go through the list of previous match cases, and accumulate a union of all types that were already
/// matched by these patterns.
fn type_excluded_by_previous_patterns<'db>(
    db: &'db dyn Db,
    mut predicate: PatternPredicate<'db>,
) -> Type<'db> {
    let mut builder = UnionBuilder::new(db);
    while let Some(previous) = predicate.previous_predicate(db) {
        predicate = *previous;

        if predicate.guard(db).is_none() {
            builder = builder.add(pattern_kind_to_type(db, predicate.kind(db)));
        }
    }
    builder.build()
}

/// Analyze a pattern predicate to determine its static truthiness.
///
/// This is a Salsa tracked function to enable memoization. Without memoization, for a match
/// statement with N cases where each case references the subject (e.g., `self`), we would
/// re-analyze each pattern O(N) times (once per reference), leading to O(N²) total work.
/// With memoization, each pattern is analyzed exactly once.
#[salsa::tracked(
    cycle_initial = |_, _, _| Truthiness::Ambiguous,
    heap_size = get_size2::GetSize::get_heap_size
)]
fn analyze_pattern_predicate<'db>(db: &'db dyn Db, predicate: PatternPredicate<'db>) -> Truthiness {
    let subject_ty = infer_expression_type(db, predicate.subject(db), TypeContext::default());

    let narrowed_subject = IntersectionBuilder::new(db)
        .add_positive(subject_ty)
        .add_negative(type_excluded_by_previous_patterns(db, predicate));

    let narrowed_subject_ty = narrowed_subject.clone().build();

    // Consider a case where we match on a subject type of `Self` with an upper bound of `Answer`,
    // where `Answer` is a {YES, NO} enum. After a previous pattern matching on `NO`, the narrowed
    // subject type is `Self & ~Literal[NO]`. This type is *not* equivalent to `Literal[YES]`,
    // because `Self` could also specialize to `Literal[NO]` or `Never`, making the intersection
    // empty. However, if the current pattern matches on `YES`, the *next* narrowed subject type
    // will be `Self & ~Literal[NO] & ~Literal[YES]`, which *is* always equivalent to `Never`. This
    // means that subsequent patterns can never match. And we know that if we reach this point,
    // the current pattern will have to match. We return `AlwaysTrue` here, since the call to
    // `analyze_single_pattern_predicate_kind` below would return `Ambiguous` in this case.
    let next_narrowed_subject_ty = narrowed_subject
        .add_negative(pattern_kind_to_type(db, predicate.kind(db)))
        .build();
    if !narrowed_subject_ty.is_never() && next_narrowed_subject_ty.is_never() {
        return Truthiness::AlwaysTrue;
    }

    let truthiness =
        analyze_single_pattern_predicate_kind(db, predicate.kind(db), narrowed_subject_ty);

    if truthiness == Truthiness::AlwaysTrue && predicate.guard(db).is_some() {
        // Fall back to ambiguous, the guard might change the result.
        // TODO: actually analyze guard truthiness
        Truthiness::Ambiguous
    } else {
        truthiness
    }
}

fn analyze_single_pattern_predicate_kind<'db>(
    db: &'db dyn Db,
    predicate_kind: &PatternPredicateKind<'db>,
    subject_ty: Type<'db>,
) -> Truthiness {
    match predicate_kind {
        PatternPredicateKind::Value(value) => {
            let value_ty = infer_expression_type(db, *value, TypeContext::default());

            if subject_ty.is_single_valued(db) {
                Truthiness::from(subject_ty.is_equivalent_to(db, value_ty))
            } else {
                Truthiness::Ambiguous
            }
        }
        PatternPredicateKind::Singleton(singleton) => {
            let singleton_ty = singleton_to_type(db, *singleton);

            if subject_ty.is_equivalent_to(db, singleton_ty) {
                Truthiness::AlwaysTrue
            } else if subject_ty.is_disjoint_from(db, singleton_ty) {
                Truthiness::AlwaysFalse
            } else {
                Truthiness::Ambiguous
            }
        }
        PatternPredicateKind::Or(predicates) => {
            use std::ops::ControlFlow;

            let mut excluded_types = vec![];
            let (ControlFlow::Break(truthiness) | ControlFlow::Continue(truthiness)) = predicates
                .iter()
                .map(|p| {
                    let narrowed_subject_ty = IntersectionBuilder::new(db)
                        .add_positive(subject_ty)
                        .add_negative(UnionType::from_elements(db, excluded_types.iter()))
                        .build();

                    excluded_types.push(pattern_kind_to_type(db, p));

                    analyze_single_pattern_predicate_kind(db, p, narrowed_subject_ty)
                })
                // this is just a "max", but with a slight optimization: `AlwaysTrue` is the "greatest" possible element, so we short-circuit if we get there
                .try_fold(Truthiness::AlwaysFalse, |acc, next| match (acc, next) {
                    (Truthiness::AlwaysTrue, _) | (_, Truthiness::AlwaysTrue) => {
                        ControlFlow::Break(Truthiness::AlwaysTrue)
                    }
                    (Truthiness::Ambiguous, _) | (_, Truthiness::Ambiguous) => {
                        ControlFlow::Continue(Truthiness::Ambiguous)
                    }
                    (Truthiness::AlwaysFalse, Truthiness::AlwaysFalse) => {
                        ControlFlow::Continue(Truthiness::AlwaysFalse)
                    }
                });
            truthiness
        }
        PatternPredicateKind::Class(class_expr, kind) => {
            let class_ty = infer_expression_type(db, *class_expr, TypeContext::default())
                .as_class_literal()
                .map(|class| Type::instance(db, class.top_materialization(db)));

            class_ty.map_or(Truthiness::Ambiguous, |class_ty| {
                if subject_ty.is_subtype_of(db, class_ty) {
                    if kind.is_irrefutable() {
                        Truthiness::AlwaysTrue
                    } else {
                        // A class pattern like `case Point(x=0, y=0)` is not irrefutable,
                        // i.e. it does not match all instances of `Point`. This means that
                        // we can't tell for sure if this pattern will match or not.
                        Truthiness::Ambiguous
                    }
                } else if subject_ty.is_disjoint_from(db, class_ty) {
                    Truthiness::AlwaysFalse
                } else {
                    Truthiness::Ambiguous
                }
            })
        }
        PatternPredicateKind::Mapping(kind) => {
            let mapping_ty = mapping_pattern_type(db);
            if subject_ty.is_subtype_of(db, mapping_ty) {
                if kind.is_irrefutable() {
                    Truthiness::AlwaysTrue
                } else {
                    Truthiness::Ambiguous
                }
            } else if subject_ty.is_disjoint_from(db, mapping_ty) {
                Truthiness::AlwaysFalse
            } else {
                Truthiness::Ambiguous
            }
        }
        PatternPredicateKind::As(pattern, _) => pattern
            .as_deref()
            .map(|p| analyze_single_pattern_predicate_kind(db, p, subject_ty))
            .unwrap_or(Truthiness::AlwaysTrue),
        PatternPredicateKind::Unsupported => Truthiness::Ambiguous,
    }
}

fn analyze_single(db: &dyn Db, predicate: &Predicate) -> Truthiness {
    let _span = tracing::trace_span!("analyze_single", ?predicate).entered();

    match predicate.node {
        PredicateNode::Expression(test_expr) => {
            infer_expression_type(db, test_expr, TypeContext::default())
                .bool(db)
                .negate_if(!predicate.is_positive)
        }
        PredicateNode::IsNonTerminalCall(CallableAndCallExpr {
            callable,
            call_expr,
            is_await,
        }) => {
            // We first infer just the type of the callable. In the most likely case that the
            // function is not marked with `NoReturn`, or that it always returns `NoReturn`,
            // doing so allows us to avoid the more expensive work of inferring the entire call
            // expression (which could involve inferring argument types to possibly run the overload
            // selection algorithm).
            // Avoiding this on the happy-path is important because these constraints can be
            // very large in number, since we add them on all statement level function calls.
            let ty = infer_expression_type(db, callable, TypeContext::default());

            // Short-circuit for well known types that are known not to return `Never` when called.
            // Without the short-circuit, we've seen that threads keep blocking each other
            // because they all try to acquire Salsa's `CallableType` lock that ensures each type
            // is only interned once. The lock is so heavily congested because there are only
            // very few dynamic types, in which case Salsa's sharding the locks by value
            // doesn't help much.
            // See <https://github.com/astral-sh/ty/issues/968>.
            if matches!(ty, Type::Dynamic(_)) {
                return Truthiness::AlwaysTrue.negate_if(!predicate.is_positive);
            }

            let overloads_iterator = if let Some(callable) = ty
                .try_upcast_to_callable(db)
                .and_then(CallableTypes::exactly_one)
            {
                callable.signatures(db).overloads.iter()
            } else {
                return Truthiness::AlwaysTrue.negate_if(!predicate.is_positive);
            };

            let mut no_overloads_return_never = true;
            let mut all_overloads_return_never = true;
            let mut any_overload_is_generic = false;

            for overload in overloads_iterator {
                let returns_never = overload.return_ty.is_equivalent_to(db, Type::Never);
                no_overloads_return_never &= !returns_never;
                all_overloads_return_never &= returns_never;
                any_overload_is_generic |= overload.return_ty.has_typevar(db);
            }

            if no_overloads_return_never && !any_overload_is_generic && !is_await {
                Truthiness::AlwaysTrue
            } else if all_overloads_return_never {
                Truthiness::AlwaysFalse
            } else {
                let call_expr_ty = infer_expression_type(db, call_expr, TypeContext::default());
                if call_expr_ty.is_equivalent_to(db, Type::Never) {
                    Truthiness::AlwaysFalse
                } else {
                    Truthiness::AlwaysTrue
                }
            }
            .negate_if(!predicate.is_positive)
        }
        PredicateNode::Pattern(inner) => analyze_pattern_predicate(db, inner),
        PredicateNode::StarImportPlaceholder(star_import) => {
            let place_table = place_table(db, star_import.scope(db));
            let symbol = place_table.symbol(star_import.symbol_id(db));
            let referenced_file = star_import.referenced_file(db);

            let requires_explicit_reexport = match dunder_all_names(db, referenced_file) {
                Some(all_names) => {
                    if all_names.contains(symbol.name()) {
                        Some(RequiresExplicitReExport::No)
                    } else {
                        tracing::trace!(
                            "Symbol `{}` (via star import) not found in `__all__` of `{}`",
                            symbol.name(),
                            referenced_file.path(db)
                        );
                        return Truthiness::AlwaysFalse;
                    }
                }
                None => None,
            };

            match imported_symbol(
                db,
                Some(referenced_file),
                symbol.name(),
                requires_explicit_reexport,
            )
            .place
            {
                crate::place::Place::Defined(crate::place::DefinedPlace {
                    definedness: crate::place::Definedness::AlwaysDefined,
                    ..
                }) => Truthiness::AlwaysTrue,
                crate::place::Place::Defined(crate::place::DefinedPlace {
                    definedness: crate::place::Definedness::PossiblyUndefined,
                    ..
                }) => Truthiness::Ambiguous,
                crate::place::Place::Undefined => Truthiness::AlwaysFalse,
            }
        }
    }
}

/// Narrow a type by walking a TDD narrowing constraint.
///
/// The TDD represents a ternary formula over predicates that encodes which predicates
/// hold along a particular control flow path. We walk from root to leaves, accumulating
/// narrowing constraints.
///
/// At each interior node, we branch based on whether the predicate is true or false:
/// - True branch: apply positive narrowing from the predicate
/// - False branch: apply negative narrowing from the predicate
///
/// The "ambiguous" branch in the TDD is not followed for narrowing purposes, because
/// narrowing constraints record which predicates hold along the control flow path.
/// The predicates may be statically ambiguous (we can't determine their truthiness
/// at analysis time), but they still hold dynamically at runtime and should be used
/// for narrowing.
///
/// At leaves:
/// - `ALWAYS_TRUE` or `AMBIGUOUS`: apply all accumulated narrowing to the base type
/// - `ALWAYS_FALSE`: this path is impossible → Never
///
/// The final result is the union of all path results.
pub(crate) fn narrow_by_constraint<'db>(
    db: &'db dyn Db,
    constraints: &ReachabilityConstraints,
    predicates: &Predicates<'db>,
    id: ScopedReachabilityConstraintId,
    base_ty: Type<'db>,
    place: ScopedPlaceId,
) -> Type<'db> {
    narrow_by_constraint_inner(db, constraints, predicates, id, base_ty, place, None)
}

/// Inner recursive helper that accumulates narrowing constraints along each TDD path.
fn narrow_by_constraint_inner<'db>(
    db: &'db dyn Db,
    constraints: &ReachabilityConstraints,
    predicates: &Predicates<'db>,
    id: ScopedReachabilityConstraintId,
    base_ty: Type<'db>,
    place: ScopedPlaceId,
    accumulated: Option<NarrowingConstraint<'db>>,
) -> Type<'db> {
    match id {
        ALWAYS_TRUE | AMBIGUOUS => {
            // Apply all accumulated narrowing constraints to the base type
            match accumulated {
                Some(constraint) => NarrowingConstraint::intersection(base_ty)
                    .merge_constraint_and(constraint)
                    .evaluate_constraint_type(db),
                None => base_ty,
            }
        }
        ALWAYS_FALSE => Type::Never,
        _ => {
            let node = constraints.get_interior_node(id);
            let predicate = predicates[node.atom];

            // `IsNonTerminalCall` predicates don't narrow any variable; they only
            // affect reachability. Evaluate the predicate to determine which
            // path(s) are reachable, rather than walking both branches.
            // `IsNonTerminalCall` always evaluates to `AlwaysTrue` or `AlwaysFalse`,
            // never `Ambiguous`.
            if matches!(predicate.node, PredicateNode::IsNonTerminalCall(_)) {
                return match analyze_single(db, &predicate) {
                    Truthiness::AlwaysTrue => narrow_by_constraint_inner(
                        db,
                        constraints,
                        predicates,
                        node.if_true,
                        base_ty,
                        place,
                        accumulated,
                    ),
                    Truthiness::AlwaysFalse => narrow_by_constraint_inner(
                        db,
                        constraints,
                        predicates,
                        node.if_false,
                        base_ty,
                        place,
                        accumulated,
                    ),
                    Truthiness::Ambiguous => {
                        unreachable!("`IsNonTerminalCall` predicates should never be Ambiguous")
                    }
                };
            }

            // Check if this predicate narrows the variable we're interested in.
            let pos_constraint = infer_narrowing_constraint(db, predicate, place);

            // If the true branch is statically unreachable, skip it entirely.
            if node.if_true == ALWAYS_FALSE {
                let neg_predicate = Predicate {
                    node: predicate.node,
                    is_positive: !predicate.is_positive,
                };
                let neg_constraint = infer_narrowing_constraint(db, neg_predicate, place);
                let false_accumulated = accumulate_constraint(accumulated, neg_constraint);
                return narrow_by_constraint_inner(
                    db,
                    constraints,
                    predicates,
                    node.if_false,
                    base_ty,
                    place,
                    false_accumulated,
                );
            }

            // If the false branch is statically unreachable, skip it entirely.
            if node.if_false == ALWAYS_FALSE {
                let true_accumulated = accumulate_constraint(accumulated, pos_constraint);
                return narrow_by_constraint_inner(
                    db,
                    constraints,
                    predicates,
                    node.if_true,
                    base_ty,
                    place,
                    true_accumulated,
                );
            }

            // True branch: predicate holds → accumulate positive narrowing
            let true_accumulated = accumulate_constraint(accumulated.clone(), pos_constraint);
            let true_ty = narrow_by_constraint_inner(
                db,
                constraints,
                predicates,
                node.if_true,
                base_ty,
                place,
                true_accumulated,
            );

            // False branch: predicate doesn't hold → accumulate negative narrowing
            let neg_predicate = Predicate {
                node: predicate.node,
                is_positive: !predicate.is_positive,
            };
            let neg_constraint = infer_narrowing_constraint(db, neg_predicate, place);
            let false_accumulated = accumulate_constraint(accumulated, neg_constraint);
            let false_ty = narrow_by_constraint_inner(
                db,
                constraints,
                predicates,
                node.if_false,
                base_ty,
                place,
                false_accumulated,
            );

            UnionType::from_two_elements(db, true_ty, false_ty)
        }
    }
}

/// Analyze the statically known reachability for a given constraint.
pub(crate) fn evaluate_reachability_constraint<'db>(
    db: &'db dyn Db,
    constraints: &ReachabilityConstraints,
    predicates: &Predicates<'db>,
    mut id: ScopedReachabilityConstraintId,
) -> Truthiness {
    loop {
        let node = match id {
            ALWAYS_TRUE => return Truthiness::AlwaysTrue,
            AMBIGUOUS => return Truthiness::Ambiguous,
            ALWAYS_FALSE => return Truthiness::AlwaysFalse,
            _ => {
                // `id` gives us the index of this node in the IndexVec that we used when
                // constructing this BDD. When finalizing the builder, we threw away any
                // interior nodes that weren't marked as used. The `used_indices` bit vector
                // lets us verify that this node was marked as used, and the rank of that bit
                // in the bit vector tells us where this node lives in the "condensed"
                // `used_interiors` vector.
                let raw_index = id.as_u32() as usize;
                debug_assert!(
                    constraints
                        .used_indices()
                        .get_bit(raw_index)
                        .unwrap_or(false),
                    "all used reachability constraints should have been marked as used",
                );
                let index = constraints.used_indices().rank(raw_index) as usize;
                constraints.used_interiors()[index]
            }
        };
        let predicate = &predicates[node.atom];
        match analyze_single(db, predicate) {
            Truthiness::AlwaysTrue => id = node.if_true,
            Truthiness::Ambiguous => id = node.if_ambiguous,
            Truthiness::AlwaysFalse => id = node.if_false,
        }
    }
}

/// Check whether a diagnostic emitted at `range` is in reachable code, considering both
/// scope reachability and statement-level reachability within the scope.
pub(crate) fn is_range_reachable<'db>(
    db: &'db dyn crate::Db,
    index: &SemanticIndex<'db>,
    scope_id: FileScopeId,
    range: TextRange,
) -> bool {
    index.ancestor_scopes(scope_id).all(|(scope_id, _)| {
        let use_def = index.use_def_map(scope_id);
        !use_def
            .range_reachability()
            .any(|(entry_range, constraint)| {
                entry_range.contains_range(range) && !is_reachable(db, use_def, constraint)
            })
    })
}

pub(crate) fn is_reachable<'db>(
    db: &'db dyn Db,
    use_def: &UseDefMap<'db>,
    reachability: ScopedReachabilityConstraintId,
) -> bool {
    evaluate_reachability(db, use_def, reachability).may_be_true()
}

pub(crate) fn binding_reachability<'db, 'map>(
    db: &'db dyn Db,
    use_def: &'map UseDefMap<'db>,
    binding: &BindingWithConstraints<'map, 'db>,
) -> Truthiness {
    evaluate_reachability_constraint(
        db,
        use_def.reachability_constraints(),
        use_def.predicates(),
        binding.reachability_constraint,
    )
}

pub(super) fn evaluate_reachability(
    db: &dyn Db,
    use_def: &UseDefMap,
    reachability: ScopedReachabilityConstraintId,
) -> Truthiness {
    evaluate_reachability_constraint(
        db,
        use_def.reachability_constraints(),
        use_def.predicates(),
        reachability,
    )
}
