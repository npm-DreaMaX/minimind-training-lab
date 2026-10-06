"""Pinned official Omni objective, with separately observable components.

Dataset labels are already shifted. Never apply another next-token shift here.
STOP receives 10x loss in the numerator, while the denominator counts positions.
"""
import torch
import torch.nn.functional as F


def omni_loss(result,text_labels,audio_labels,diagnostics=False):
    text_raw=F.cross_entropy(result.logits.flatten(0,1),text_labels.flatten(),reduction='none')
    text_valid=text_labels.flatten()!=-100
    text_count=text_valid.sum()
    text_ce=(text_raw*text_valid).sum()/(text_count+1e-9)
    channels=[]; counts=[]; stop_counts=[]; stop_nll=[]; stop_correct=[]
    sample_audio_sum=[]; sample_audio_count=[]; sample_stop_count=[]; sample_stop_correct=[]
    for i,logits in enumerate(result.audio_logits):
        labels=audio_labels[:,i].reshape(-1)
        raw=F.cross_entropy(logits.flatten(0,1),labels,reduction='none')
        valid=labels!=-100; stop=labels==2050; n=valid.sum()
        channels.append((raw*valid*(1+9*stop)).sum()/(n+1e-9))
        counts.append(n); stop_counts.append(stop.sum())
        if diagnostics:
            stop_nll.append((raw.detach()*stop).sum())
            correct_stop=(logits.detach().argmax(-1).flatten()==2050)&stop
            stop_correct.append(correct_stop.sum())
            batch_size=text_labels.shape[0]
            sample_audio_sum.append((raw.detach()*valid*(1+9*stop)).reshape(batch_size,-1).sum(1))
            sample_audio_count.append(valid.reshape(batch_size,-1).sum(1))
            sample_stop_count.append(stop.reshape(batch_size,-1).sum(1))
            sample_stop_correct.append(correct_stop.reshape(batch_size,-1).sum(1))
    if len(channels)!=8: raise ValueError('Expected eight audio codebook heads')
    audio_ce=torch.stack(channels).mean()
    result_dict={'loss':text_ce+audio_ce+result.aux_loss,'text_ce':text_ce,'audio_ce':audio_ce,
                 'audio_codebook_ce':torch.stack(channels),'aux_loss':result.aux_loss,
                 'text_count':text_count,'audio_counts':torch.stack(counts),'stop_counts':torch.stack(stop_counts)}
    if diagnostics:
        result_dict.update(stop_nll_sum=torch.stack(stop_nll),stop_correct=torch.stack(stop_correct))
        result_dict['per_sample']={
            'text_nll_sum':(text_raw.detach()*text_valid).reshape(text_labels.shape[0],-1).sum(1),
            'text_count':text_valid.reshape(text_labels.shape[0],-1).sum(1),
            'audio_weighted_nll_sum':torch.stack(sample_audio_sum,dim=1),
            'audio_count':torch.stack(sample_audio_count,dim=1),
            'stop_count':torch.stack(sample_stop_count,dim=1),
            'stop_correct':torch.stack(sample_stop_correct,dim=1)}
    return result_dict
