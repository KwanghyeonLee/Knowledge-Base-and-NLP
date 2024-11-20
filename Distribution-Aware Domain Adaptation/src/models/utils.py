from typing import List, Union

import transformers
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.datamodule.utils import get_terms, remove_special_tokens


def get_scheduler(
    optimizer, scheduler: str, warmup_steps: int, t_total: int, num_devices: Union[List[int], int]
):
    """Returns the correct learning rate scheduler.

    Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine,
    warmupcosinewithhardrestarts
    """
    total_devices = num_devices if isinstance(num_devices, int) else len(num_devices)
    warmup_steps //= total_devices
    t_total //= total_devices
    scheduler = scheduler.lower()
    schedulers = {
        "constantlr": transformers.get_constant_schedule(optimizer),
        "warmupconstant": transformers.get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps
        ),
        "warmuplinear": transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        ),
        "warmupcosine": transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        ),
        "warmupcosinewithhardrestarts": transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        ),
        "reduceonplateau": ReduceLROnPlateau(
            optimizer, mode="min", patience=warmup_steps // 10, threshold=0.1
        ),
    }
    try:
        return schedulers[scheduler]
    except Exception as e:
        raise ValueError(f"Unknown scheduler {scheduler}")


def convert2readable(
    tokenizer,
    pseudo_query_logit: Tensor,
    query_logit: Tensor,
    extraction_task_prefix: str,
    doc_logit: Tensor,
    ids: List[str] = None,
    is_train: bool = False,
):
    def convert_logit2text(logit):
        return list(
            map(
                lambda txt: remove_special_tokens(txt, tokenizer.all_special_tokens),
                tokenizer.batch_decode(logit),
            )
        )

    def convert_text2terms(texts):
        return list(map(lambda text: get_terms(text), texts))

    pseudo_query_text: List[str] = convert_logit2text(pseudo_query_logit)
    pseudo_query_terms: List[List[str]] = convert_text2terms(pseudo_query_text)

    true_query_text: List[str] = convert_logit2text(query_logit)
    true_query_terms: List[List[str]] = convert_text2terms(true_query_text)

    if is_train:
        docs_text = list(
            zip(
                *map(
                    #
                    lambda idx: list(
                        map(
                            lambda txt: txt.replace(f"{extraction_task_prefix}: ", ""),
                            convert_logit2text(doc_logit[:, (128 * idx) : (128 * (idx + 1) - 1)]),
                        )
                    ),
                    range(0, 4),
                )
            )
        )
        docs_terms = list(map(lambda batch: convert_text2terms(batch), docs_text))

    else:
        docs_text = convert_logit2text(doc_logit)
        docs_terms = list(
            map(
                lambda sent: get_terms(sent.replace(f"{extraction_task_prefix}: ", "")),
                docs_text,
            )
        )

    def concat_list(batch_list):
        return list(
            map(
                lambda nested_list: list(map(lambda toks: ", ".join(toks), nested_list)),
                batch_list,
            )
        )

    save_target = (
        list(
            zip(
                ids,
                pseudo_query_text,
            )
        )
        if ids
        else None
    )

    return docs_terms, true_query_terms, pseudo_query_terms, save_target
