"""Explicit, fail-closed state transfer; no silent missing-key initialization."""
import torch


def transfer_ar_weights(target, source, kind):
    """Load a validated AR state into a Hybrid or official Omni instance.

    The caller constructs the target under a recorded seed. New parameters retain
    that constructor initialization. Attention algorithms are not interchangeable
    merely because some matrix dimensions happen to match.
    """
    target_state=target.state_dict()
    if kind=='hybrid':
        replaced=[i for i,t in enumerate(target.config.layer_types) if t=='linear_attention']
        permitted_new=lambda k:any(k.startswith(f'model.layers.{i}.linear_attn.') for i in replaced)
        permitted_discard=lambda k:any(k.startswith(f'model.layers.{i}.self_attn.') for i in replaced)
    elif kind=='omni':
        permitted_new=lambda k:k.startswith(('talker.','audio_proj.','vision_proj.'))
        permitted_discard=lambda k:False
    else: raise ValueError(kind)
    missing=sorted(set(target_state)-set(source)); discarded=sorted(set(source)-set(target_state))
    unexpected_new=[k for k in missing if not permitted_new(k)]
    unexpected_discard=[k for k in discarded if not permitted_discard(k)]
    if unexpected_new or unexpected_discard:
        raise ValueError({'unexpected_new':unexpected_new,'unexpected_discard':unexpected_discard})
    copied=[]
    for name,value in source.items():
        if not torch.isfinite(value).all(): raise ValueError(f'Nonfinite source: {name}')
        if name in target_state and value.shape!=target_state[name].shape:
            raise ValueError(f'Shape mismatch: {name}, source={value.shape}, target={target_state[name].shape}')
    with torch.no_grad():
        for name,value in source.items():
            if name not in target_state: continue
            target_state[name].copy_(value); copied.append(name)
    duplicated=[]
    if kind=='omni':
        if target.config.talker_hidden_size!=target.config.hidden_size:
            raise ValueError('This transfer reproduces official equal-width Thinker-to-Talker initialization only')
        n=len(target.talker.layers); offset=len(target.thinker.layers)-n
        if offset<0: raise ValueError('Talker has more layers than Thinker')
        for i,block in enumerate(target.talker.layers):
            block.load_state_dict(target.thinker.layers[offset+i].state_dict(),strict=True)
            duplicated.extend({'target':f'talker.layers.{i}.{k}','source':f'model.layers.{offset+i}.{k}'} for k in block.state_dict())
    duplicate_names={x['target'] for x in duplicated}
    random_names=[k for k in missing if k not in duplicate_names]
    return {'kind':kind,'copied_keys':copied,'new_keys':missing,'discarded_keys':discarded,
            'duplicated_from_thinker':duplicated,'constructor_initialized_keys':random_names,
            'copied_unique_parameters':sum(p.numel() for name,p in target.named_parameters() if name in copied),
            'new_unique_parameters':sum(p.numel() for name,p in target.named_parameters() if name in missing),
            'duplicated_unique_parameters':sum(p.numel() for name,p in target.named_parameters() if name in duplicate_names),
            'constructor_initialized_unique_parameters':sum(p.numel() for name,p in target.named_parameters() if name in random_names),
            'total_unique_parameters':sum(p.numel() for p in target.parameters()),
            'policy':'Exact shared names and shapes; allowlisted structural changes; official Thinker last layers copied to Omni Talker; new modules retain recorded seeded constructor initialization'}
