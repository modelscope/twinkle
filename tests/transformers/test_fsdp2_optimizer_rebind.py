# Copyright (c) ModelScope Contributors. All rights reserved.
"""FSDP2 leaves the parameters outside every ``fully_shard`` unit stale in the optimizer.

Casting a parameter to the mixed-precision compute dtype replaces the ``Parameter`` object, and
``accelerator.prepare`` rebuilds the optimizer over the sharded parameters -- but a vision tower and
the embedding/tail that accelerate carves into their own units keep their PRE-WRAP objects in the
optimizer's ``param_groups``. Training steps the model's objects, so nothing looks wrong until a
checkpoint asks torch to map optimizer state onto parameter FQNs by object identity and the save
dies after the whole run has trained. ``_rebind_unsharded_optimizer_params`` closes that gap at
wrap time, where no optimizer state exists yet.

These cases attack the pairing rule rather than confirming it: the replacement set must be the
model's plain trainable parameters, a shape occurring twice must pair by traversal order, and a
shape multiset that does not match must leave the optimizer untouched so the loud save-time refusal
still fires instead of a silent guess.
"""
import torch

from twinkle.model.transformers.strategy.accelerate import AccelerateStrategy

rebind = AccelerateStrategy._rebind_unsharded_optimizer_params


def _optimizer(*params):
    return torch.optim.AdamW(list(params), lr=1e-5)


def _params_of(optimizer):
    return [param for group in optimizer.param_groups for param in group['params']]


def _same(actual, expected):
    """Identity comparison: ``==`` on a list of tensors is elementwise and raises."""
    return len(actual) == len(expected) and all(a is e for a, e in zip(actual, expected))



class _Model(torch.nn.Module):
    """A stand-in for the wrapped model: named parameters in a declared order."""

    def __init__(self, named):
        super().__init__()
        for index, (_, param) in enumerate(named):
            self.register_parameter(f'p{index}', param)
        self._names = [name for name, _ in named]

    def named_parameters(self, *args, **kwargs):
        return list(zip(self._names, [getattr(self, f'p{i}') for i in range(len(self._names))]))


def test_a_stale_unsharded_parameter_is_repointed_onto_the_models_object():
    """The defect itself: the optimizer names an object the wrapped model no longer exposes."""
    current = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16))
    stale = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float32))
    model = _Model([('model.visual.patch_embed.proj.weight', current)])
    optimizer = _optimizer(stale)

    rebind(model, optimizer)

    assert _params_of(optimizer) == [current], 'the optimizer must step the model\'s own parameter'
    assert all(param is not stale for param in _params_of(optimizer))


def test_a_shape_occurring_twice_pairs_by_traversal_order():
    """An untied embedding and lm_head share a shape, so order is the only unambiguous pairing."""
    embed_new = torch.nn.Parameter(torch.zeros(6, 2))
    head_new = torch.nn.Parameter(torch.ones(6, 2))
    embed_old = torch.nn.Parameter(torch.zeros(6, 2, dtype=torch.float32))
    head_old = torch.nn.Parameter(torch.ones(6, 2, dtype=torch.float32))
    model = _Model([('model.embed_tokens.weight', embed_new), ('lm_head.weight', head_new)])
    optimizer = _optimizer(embed_old, head_old)

    rebind(model, optimizer)

    assert _params_of(optimizer) == [embed_new, head_new], 'same-shape offenders must pair in order'


def test_a_mismatched_shape_multiset_leaves_the_optimizer_untouched():
    """No provable pairing -> no rebind, so the save-time guard still refuses loudly."""
    current = torch.nn.Parameter(torch.zeros(4, 3))
    stale = torch.nn.Parameter(torch.zeros(9, 9, dtype=torch.float32))
    model = _Model([('weight', current)])
    optimizer = _optimizer(stale)

    rebind(model, optimizer)

    assert _params_of(optimizer) == [stale], 'an unprovable pairing must not be guessed'


def test_a_count_mismatch_leaves_the_optimizer_untouched():
    """Two stale objects against one plain model parameter of that shape: ambiguous, so no rebind."""
    current = torch.nn.Parameter(torch.zeros(4, 3))
    stale_a = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float32))
    stale_b = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float32))
    model = _Model([('weight', current)])
    optimizer = _optimizer(stale_a, stale_b)

    rebind(model, optimizer)

    assert _params_of(optimizer) == [stale_a, stale_b]


def test_a_frozen_model_parameter_is_not_offered_as_a_replacement():
    """``plain`` is filtered on requires_grad: a frozen weight cannot own an optimizer moment."""
    trainable = torch.nn.Parameter(torch.zeros(4, 3))
    frozen = torch.nn.Parameter(torch.zeros(4, 3))
    frozen.requires_grad_(False)
    stale = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float32))
    # The frozen one is declared FIRST, so an unfiltered pairing would hand the stale object to it.
    model = _Model([('frozen.weight', frozen), ('trainable.weight', trainable)])
    optimizer = _optimizer(stale)

    rebind(model, optimizer)

    assert _params_of(optimizer) == [trainable], 'the replacement must be the trainable parameter'


def test_an_already_reachable_optimizer_is_left_alone():
    """Control for the above: nothing to do when the optimizer already names the model's objects."""
    current = torch.nn.Parameter(torch.zeros(4, 3))
    model = _Model([('weight', current)])
    optimizer = _optimizer(current)

    rebind(model, optimizer)

    assert _params_of(optimizer) == [current]


def test_rebinding_preserves_every_other_group_and_hyperparameter():
    """The repair touches only the parameter objects, not the groups they sit in."""
    current = torch.nn.Parameter(torch.zeros(4, 3))
    stale = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float32))
    other = torch.nn.Parameter(torch.zeros(2, 2))
    model = _Model([('weight', current), ('other', other)])
    optimizer = torch.optim.AdamW([{
        'params': [stale],
        'lr': 3e-5,
        'weight_decay': 0.1
    }, {
        'params': [other],
        'lr': 1e-5
    }])

    rebind(model, optimizer)

    assert optimizer.param_groups[0]['lr'] == 3e-5
    assert optimizer.param_groups[0]['weight_decay'] == 0.1
    assert optimizer.param_groups[0]['params'] == [current]
    assert optimizer.param_groups[1]['params'] == [other]


def test_an_optimizer_without_param_groups_does_not_crash():
    """A forward-only build carries no optimizer parameters; the repair must be a silent no-op.

    torch's own optimizers refuse an empty parameter list, so the stand-in is the shape the strategy
    actually guards against: an object whose ``param_groups`` is empty.
    """
    class _NoGroups:
        param_groups = []

    model = _Model([('weight', torch.nn.Parameter(torch.zeros(2)))])

    rebind(model, _NoGroups())

    assert _NoGroups.param_groups == []


def test_a_plain_parameter_the_wrap_already_rebound_is_not_a_replacement_candidate():
    """The real failure shape: 12 unsharded model parameters against 11 stale optimizer objects.

    One unsharded parameter was already repointed by the wrap, so it sits in the optimizer and is
    reachable. Offering it as a replacement for a different offender would hand one weight two
    optimizer entries and leave the real offender stale, so the candidate set excludes everything
    the optimizer already names -- which is what makes the per-shape counts come out one-for-one.
    """
    rebound = torch.nn.Parameter(torch.zeros(8))          # plain, already in the optimizer
    visual_new = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16))
    visual_old = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float32))
    model = _Model([('model.norm.weight', rebound), ('model.visual.proj.weight', visual_new)])
    optimizer = _optimizer(rebound, visual_old)

    rebind(model, optimizer)

    assert _params_of(optimizer) == [rebound, visual_new]


def test_a_repair_after_training_carries_the_adam_moments_onto_the_new_object():
    """The save-time repair: the optimizer already holds state, keyed by the stale object.

    FSDP2's mixed precision can replace a parameter object AFTER prepare, so the wrap-time repair
    finds nothing to do and the divergence is only observable at save time -- by which point Adam has
    moments. An offender is a parameter FSDP2 left unsharded, so it is full-shaped on both sides and
    the moments transfer by identity; leaving them keyed to the stale object would orphan them and
    the resumed run would restart its second-moment estimate from zero.
    """
    current = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16))
    stale = torch.nn.Parameter(torch.zeros(4, 3, dtype=torch.float32))
    model = _Model([('model.visual.proj.weight', current)])
    optimizer = _optimizer(stale)
    optimizer.state[stale] = {'step': 7, 'exp_avg': torch.ones(4, 3), 'exp_avg_sq': torch.full((4, 3), 2.0)}

    rebind(model, optimizer)

    assert _params_of(optimizer) == [current]
    assert all(key is not stale for key in optimizer.state)
    assert optimizer.state[current]['step'] == 7
    assert torch.equal(optimizer.state[current]['exp_avg'], torch.ones(4, 3))
    assert torch.equal(optimizer.state[current]['exp_avg_sq'], torch.full((4, 3), 2.0))


def test_an_unprovable_repair_leaves_the_state_keyed_as_it_was():
    """Control: when no bucket matches, neither the groups nor the state may be touched."""
    current = torch.nn.Parameter(torch.zeros(4, 3))
    stale = torch.nn.Parameter(torch.zeros(9, 9, dtype=torch.float32))
    model = _Model([('weight', current)])
    optimizer = _optimizer(stale)
    optimizer.state[stale] = {'step': 3}

    rebind(model, optimizer)

    assert _params_of(optimizer) == [stale]
    assert optimizer.state[stale] == {'step': 3}


# ---------------------------------------------------------------------------------------------
# The name map: prepare breaks a tie, and a later cast replaces BOTH of the objects it produced.
# ---------------------------------------------------------------------------------------------


def _named_map(model, optimizer):
    return AccelerateStrategy._remember_optimizer_params(_stub_strategy(), model, optimizer)


def _stub_strategy():
    return object.__new__(AccelerateStrategy)


def test_a_tie_broken_by_prepare_is_repaired_by_name_not_by_shape():
    """Defect 18's last offender: one stale entry, two same-shaped candidates.

    ``embed_tokens.weight`` and ``lm_head.weight`` start tied, so ``prepare`` sees one object under
    one name; it then exposes them as two, and the cast that follows replaces both. The optimizer is
    left with one stale object of that shape and no way to choose by shape -- the name recorded while
    the two sides still agreed is what makes the choice exact.
    """
    embed = torch.nn.Parameter(torch.zeros(6, 2, dtype=torch.float32))
    head = torch.nn.Parameter(torch.zeros(6, 2, dtype=torch.float32))
    stale = torch.nn.Parameter(torch.zeros(6, 2, dtype=torch.float32))
    model = _Model([('model.language_model.embed_tokens.weight', embed), ('lm_head.weight', head)])
    optimizer = _optimizer(stale)

    # By shape alone the rule must refuse: 1 offender against 2 candidates is unprovable.
    rebind(model, optimizer)
    assert _same(_params_of(optimizer), [stale]), 'an ambiguous shape must not be guessed at'

    rebind(model, optimizer, {'unused_id': 'x', id(stale): 'lm_head.weight'})
    assert _same(_params_of(optimizer), [head]), 'the recorded name decides which of the two it was'


def test_a_name_that_the_optimizer_already_holds_is_not_bound_a_second_time():
    """Rebinding onto an already-held weight would train it twice and leave the other untrained."""
    held = torch.nn.Parameter(torch.zeros(6, 2))
    other = torch.nn.Parameter(torch.zeros(6, 2))
    stale = torch.nn.Parameter(torch.zeros(6, 2))
    model = _Model([('a', held), ('b', other)])
    optimizer = _optimizer(held, stale)

    rebind(model, optimizer, {id(stale): 'a'})

    # The name is rejected because 'a' is already held, and the shape rule then proves the pairing
    # it can: 'b' is plain, trainable and unnamed, and it is the only parameter of that shape.
    assert _same(_params_of(optimizer), [held, other]), 'a rejected name must not duplicate an entry'


def test_a_name_pointing_at_a_sharded_or_frozen_parameter_is_not_used():
    """The target must be a weight the loop steps: plain and trainable."""
    from torch.distributed.tensor import DTensor  # noqa: F401  (only the frozen half is exercised)

    frozen = torch.nn.Parameter(torch.zeros(6, 2), requires_grad=False)
    stale = torch.nn.Parameter(torch.zeros(6, 2))
    model = _Model([('frozen', frozen)])
    optimizer = _optimizer(stale)

    rebind(model, optimizer, {id(stale): 'frozen'})

    assert _same(_params_of(optimizer), [stale])


def test_a_stale_entry_missing_from_the_name_map_falls_through_to_the_shape_rule():
    """The name map is an addition, not a replacement: unnamed entries keep the old path."""
    current = torch.nn.Parameter(torch.zeros(4, 3))
    stale = torch.nn.Parameter(torch.zeros(4, 3))
    model = _Model([('model.visual.patch_embed.proj.weight', current)])
    optimizer = _optimizer(stale)

    rebind(model, optimizer, {})

    assert _same(_params_of(optimizer), [current]), 'the shape rule still repairs what it can prove'


# ---------------------------------------------------------------------------------------------
# The container accelerate actually returns.
# ---------------------------------------------------------------------------------------------


class _Accelerator:
    process_index = 0

    def __init__(self, prepared):
        self._prepared = prepared

    def prepare(self, *args):
        return self._prepared


def _wrap(prepared):
    strategy = object.__new__(AccelerateStrategy)
    strategy.accelerator = _Accelerator(prepared)
    model = prepared[0] if isinstance(prepared, (list, tuple)) else prepared
    return strategy, strategy.wrap_model(model, *([] if not isinstance(prepared, (list, tuple)) else prepared[1:]))


def test_wrap_model_repairs_when_prepare_returns_a_list():
    """``Accelerator.prepare`` ends with ``return result if len(result) > 1 else result[0]`` over a
    list, so a tuple test silently skipped the wrap-time repair entirely."""
    current = torch.nn.Parameter(torch.zeros(4, 3))
    stale = torch.nn.Parameter(torch.zeros(4, 3))
    model = _Model([('w', current)])
    optimizer = _optimizer(stale)

    strategy, result = _wrap([model, optimizer])

    assert result == [model, optimizer], 'the prepared objects must be handed back untouched'
    assert _same(_params_of(optimizer), [current])
    assert strategy.__dict__['_optimizer_param_names'] == {id(current): 'w'}


def test_wrap_model_accepts_a_bare_model_from_prepare():
    """One argument comes back bare; indexing it as a sequence would raise."""
    model = _Model([('w', torch.nn.Parameter(torch.zeros(4, 3)))])

    strategy, result = _wrap(model)

    assert result is model
    assert '_optimizer_param_names' not in strategy.__dict__


def test_wrap_model_still_repairs_a_tuple():
    current = torch.nn.Parameter(torch.zeros(4, 3))
    stale = torch.nn.Parameter(torch.zeros(4, 3))
    model = _Model([('w', current)])
    optimizer = _optimizer(stale)

    _, result = _wrap((model, optimizer))

    assert _same(_params_of(optimizer), [current])
    assert isinstance(result, tuple)


def test_remember_records_only_the_parameters_the_optimizer_holds():
    trained = torch.nn.Parameter(torch.zeros(4, 3))
    untrained = torch.nn.Parameter(torch.zeros(4, 3))
    model = _Model([('trained', trained), ('untrained', untrained)])
    optimizer = _optimizer(trained)

    names = _named_map(model, optimizer)

    assert names == {id(trained): 'trained'}


# ---------------------------------------------------------------------------------------------
# Which artifact counts as a saved optimizer.
# ---------------------------------------------------------------------------------------------


def test_optimizer_checkpoint_exists_follows_the_strategies_own_layout(tmp_path):
    from twinkle.model.transformers.strategy.deepspeed import DeepSpeedStrategy

    default = _stub_strategy()  # AccelerateStrategy inherits the base's optimizer.pt layout
    deepspeed = object.__new__(DeepSpeedStrategy)

    plain = tmp_path / 'plain'
    plain.mkdir()
    (plain / 'optimizer.pt').write_bytes(b'x')
    assert default.optimizer_checkpoint_exists(str(plain)) is True
    assert deepspeed.optimizer_checkpoint_exists(str(plain)) is False

    empty = tmp_path / 'empty'
    empty.mkdir()
    assert default.optimizer_checkpoint_exists(str(empty)) is False
    assert deepspeed.optimizer_checkpoint_exists(str(empty)) is False

    tagged = tmp_path / 'tagged'
    (tagged / 'global_step15').mkdir(parents=True)
    (tagged / 'latest').write_text('global_step15\n')
    assert deepspeed.optimizer_checkpoint_exists(str(tagged)) is True
    assert default.optimizer_checkpoint_exists(str(tagged)) is False

    dangling = tmp_path / 'dangling'
    dangling.mkdir()
    (dangling / 'latest').write_text('global_step15')
    assert deepspeed.optimizer_checkpoint_exists(str(dangling)) is False, 'the tag must name a real dir'

    blank = tmp_path / 'blank'
    blank.mkdir()
    (blank / 'latest').write_text('   \n')
    assert deepspeed.optimizer_checkpoint_exists(str(blank)) is False


def test_the_name_repair_carries_the_moments_so_state_dict_does_not_die():
    """A repointed group with an un-migrated moment kills ``Optimizer.state_dict()``.

    torch remaps every state key through the parameter groups, so a moment left under the stale
    object raises a bare ``KeyError: <id>`` -- which is what a full-parameter FSDP2 run hit at its
    final save once the name rule settled the tie but the migration lived only in the shape rule.
    """
    current = torch.nn.Parameter(torch.zeros(6, 2))
    head = torch.nn.Parameter(torch.zeros(6, 2))
    stale = torch.nn.Parameter(torch.zeros(6, 2))
    model = _Model([('embed', current), ('lm_head', head)])
    optimizer = _optimizer(stale)
    optimizer.state[stale] = {'step': 3, 'exp_avg': torch.ones(6, 2), 'exp_avg_sq': torch.ones(6, 2)}

    rebind(model, optimizer, {id(stale): 'embed'})

    assert _same(_params_of(optimizer), [current])
    assert all(key is current for key in optimizer.state), 'the moment must follow the parameter'
    assert optimizer.state_dict()['state'][0]['step'] == 3, 'and survive the remap through the groups'


def test_the_shape_repair_and_the_name_repair_share_one_migration():
    """Both rules must feed the same state migration, not just whichever one settles last."""
    named_new = torch.nn.Parameter(torch.zeros(6, 2))
    tied = torch.nn.Parameter(torch.zeros(6, 2))
    shaped_new = torch.nn.Parameter(torch.zeros(3, 4))
    stale_named = torch.nn.Parameter(torch.zeros(6, 2))
    stale_shaped = torch.nn.Parameter(torch.zeros(3, 4))
    model = _Model([('embed', named_new), ('lm_head', tied), ('vision', shaped_new)])
    optimizer = _optimizer(stale_named, stale_shaped)
    for stale in (stale_named, stale_shaped):
        optimizer.state[stale] = {'step': 7}

    rebind(model, optimizer, {id(stale_named): 'embed'})

    assert _same(_params_of(optimizer), [named_new, shaped_new])
    assert all(key is not stale_named and key is not stale_shaped for key in optimizer.state)
    assert optimizer.state_dict()['state'] == {0: {'step': 7}, 1: {'step': 7}}


def test_the_guard_refuses_a_moment_whose_parameter_left_every_group():
    """The same unmappable state, reached from the other side: no offender, only an orphan."""
    import pytest

    current = torch.nn.Parameter(torch.zeros(6, 2))
    orphan = torch.nn.Parameter(torch.zeros(6, 2))
    model = _Model([('w', current)])
    optimizer = _optimizer(current)
    optimizer.state[orphan] = {'step': 1}

    with pytest.raises(RuntimeError, match='no group holds'):
        AccelerateStrategy._check_optimizer_params_reachable(model, optimizer)


# ---------------------------------------------------------------------------------------------
# Sharding unties a shared weight; the duplicate must still be trained.
# ---------------------------------------------------------------------------------------------


def _duplicates(named):
    class _Dup(torch.nn.Module):
        def named_parameters(self, *args, **kwargs):
            return list(named)

    return AccelerateStrategy._duplicate_names(_Dup())


def test_duplicate_names_finds_the_name_a_tie_is_hiding():
    shared = torch.nn.Parameter(torch.zeros(6, 2))
    other = torch.nn.Parameter(torch.zeros(3))
    named = [('embed', shared), ('blocks.0.w', other), ('lm_head', shared)]

    assert _duplicates(named) == {'lm_head': 'embed'}


def test_an_untied_duplicate_joins_the_group_that_holds_its_partner():
    """The defect: the head the forward pass uses stays at its pretrained value."""
    embed = torch.nn.Parameter(torch.zeros(6, 2))
    head = torch.nn.Parameter(torch.zeros(6, 2))
    vision = torch.nn.Parameter(torch.zeros(3))
    model = _Model([('embed', embed), ('lm_head', head), ('vision', vision)])
    decay, no_decay = {'params': [embed], 'weight_decay': 0.1}, {'params': [vision], 'weight_decay': 0.0}
    optimizer = torch.optim.AdamW([decay, no_decay], lr=1e-5)

    AccelerateStrategy._adopt_untied_duplicates(model, optimizer, {'lm_head': 'embed'})

    assert decay['params'] == [embed, head], 'the head must train beside its partner'
    assert no_decay['params'] == [vision]
    assert optimizer.param_groups[0]['weight_decay'] == 0.1, 'and inherit the partner\'s weight decay'


def test_a_frozen_or_already_held_duplicate_is_not_adopted():
    embed = torch.nn.Parameter(torch.zeros(6, 2))
    frozen_head = torch.nn.Parameter(torch.zeros(6, 2), requires_grad=False)
    model = _Model([('embed', embed), ('lm_head', frozen_head)])
    optimizer = _optimizer(embed)

    AccelerateStrategy._adopt_untied_duplicates(model, optimizer, {'lm_head': 'embed'})

    assert _same(_params_of(optimizer), [embed])


def test_an_untied_model_leaves_the_optimizer_alone():
    embed = torch.nn.Parameter(torch.zeros(6, 2))
    model = _Model([('embed', embed)])
    optimizer = _optimizer(embed)

    AccelerateStrategy._adopt_untied_duplicates(model, optimizer, {})

    assert _same(_params_of(optimizer), [embed])


def test_wrap_model_adopts_the_duplicate_prepare_untied():
    """End to end through the seam: the model goes in tied and comes out untied.

    The optimizer was built over the tied view, so it holds one entry for the pair; prepare then
    exposes two objects. Without the adoption the second one never trains, and the recorded name map
    -- which the save-time repair depends on -- would not cover it either.
    """
    shared = torch.nn.Parameter(torch.zeros(6, 2))
    tied = _Model([('embed', shared), ('lm_head', shared)])
    embed = torch.nn.Parameter(torch.zeros(6, 2))
    head = torch.nn.Parameter(torch.zeros(6, 2))
    untied = _Model([('embed', embed), ('lm_head', head)])
    optimizer = _optimizer(embed)

    strategy = object.__new__(AccelerateStrategy)

    class _Untier:
        process_index = 0

        def prepare(self, model, *args):
            return [untied, optimizer]

    strategy.accelerator = _Untier()
    strategy.wrap_model(tied, optimizer)

    assert _same(_params_of(optimizer), [embed, head])
    assert strategy.__dict__['_optimizer_param_names'] == {id(embed): 'embed', id(head): 'lm_head'}
