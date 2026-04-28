import json
import logging
import os

import torch
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


MAVEN_TYPES = [
    "Know", "Warning", "Catastrophe", "Placing", "Causation", "Arriving",
    "Sending", "Protest", "Preventing_or_letting", "Motion", "Damaging",
    "Destroying", "Death", "Perception_active", "Presence", "Influence",
    "Receiving", "Check", "Hostile_encounter", "Killing", "Conquering",
    "Releasing", "Attack", "Earnings_and_losses", "Choosing", "Traveling",
    "Recovering", "Using", "Coming_to_be", "Cause_to_be_included",
    "Process_start", "Change_event_time", "Reporting", "Bodily_harm",
    "Suspicion", "Statement", "Cause_change_of_position_on_a_scale",
    "Coming_to_believe", "Expressing_publicly", "Request", "Control",
    "Supporting", "Defending", "Building", "Military_operation",
    "Self_motion", "GetReady", "Forming_relationships", "Becoming_a_member",
    "Action", "Removing", "Surrendering", "Agree_or_refuse_to_act",
    "Participation", "Deciding", "Education_teaching", "Emptying", "Getting",
    "Besieging", "Creating", "Process_end", "Body_movement", "Expansion",
    "Telling", "Change", "Legal_rulings", "Bearing_arms", "Giving",
    "Name_conferral", "Arranging", "Use_firearm", "Committing_crime",
    "Assistance", "Surrounding", "Quarreling", "Expend_resource",
    "Motion_directional", "Bringing", "Communication", "Containing",
    "Manufacturing", "Social_event", "Robbery", "Competition", "Writing",
    "Rescuing", "Judgment_communication", "Change_tool", "Hold",
    "Being_in_operation", "Recording", "Carry_goods", "Cost", "Departing",
    "GiveUp", "Change_of_leadership", "Escaping", "Aiming", "Hindering",
    "Preserving", "Create_artwork", "Openness", "Connect", "Reveal_secret",
    "Response", "Scrutiny", "Lighting", "Criminal_investigation",
    "Hiding_objects", "Confronting_problem", "Renting", "Breathing",
    "Patrolling", "Arrest", "Convincing", "Commerce_sell", "Cure",
    "Temporary_stay", "Dispersal", "Collaboration", "Extradition",
    "Change_sentiment", "Commitment", "Commerce_pay", "Filling", "Becoming",
    "Achieve", "Practice", "Cause_change_of_strength", "Supply",
    "Cause_to_amalgamate", "Scouring", "Violence", "Reforming_a_system",
    "Come_together", "Wearing", "Cause_to_make_progress", "Legality",
    "Employment", "Rite", "Publishing", "Adducing", "Exchange",
    "Ratification", "Sign_agreement", "Commerce_buy", "Imposing_obligation",
    "Rewards_and_punishments", "Institutionalization", "Testing", "Ingestion",
    "Labeling", "Kidnapping", "Submitting_documents", "Prison", "Justifying",
    "Emergency", "Terrorism", "Vocalizations", "Risk", "Resolve_problem",
    "Revenge", "Limiting", "Research", "Having_or_lacking_access", "Theft",
    "Incident", "Award",
]


def get_labels(path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    labels = ["O"]
    for event_type in MAVEN_TYPES:
        labels.append("B-" + event_type)
        labels.append("I-" + event_type)
    return labels


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.jsonl".format(mode))
    if mode == "dev" and not os.path.exists(file_path):
        file_path = os.path.join(data_dir, "valid.jsonl")

    examples = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for line in fin:
            doc = json.loads(line)
            words = [sentence["tokens"] for sentence in doc["content"]]
            labels = [["O" for _ in sentence] for sentence in words]

            if mode != "test":
                for event in doc.get("events", []):
                    event_type = event["type"]
                    for mention in event.get("mention", []):
                        sent_id = mention["sent_id"]
                        start, end = mention["offset"]
                        labels[sent_id][start] = "B-" + event_type
                        for idx in range(start + 1, end):
                            labels[sent_id][idx] = "I-" + event_type
                for mention in doc.get("negative_triggers", []):
                    sent_id = mention["sent_id"]
                    start, end = mention["offset"]
                    for idx in range(start, end):
                        labels[sent_id][idx] = "O"

            for sent_idx, sent_words in enumerate(words):
                examples.append(
                    InputExample(
                        guid="%s-%s-%d" % (mode, doc["id"], sent_idx),
                        words=sent_words,
                        labels=labels[sent_idx],
                    )
                )
    return examples


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    label_map = {label: i for i, label in enumerate(label_list)}
    assert pad_token_label_id not in label_map.values()

    features = []
    for ex_index, example in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]
            tokens.extend(word_tokens)
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: max_seq_length - special_tokens_count]
            label_ids = label_ids[: max_seq_length - special_tokens_count]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
            )
        )
    return features


def to_crf_pad(org_array, org_mask, pad_label_id):
    crf_array = [row[mask] for row, mask in zip(org_array, org_mask)]
    crf_array = pad_sequence(crf_array, batch_first=True, padding_value=pad_label_id)
    crf_pad = crf_array != pad_label_id
    crf_array[~crf_pad] = 0
    return crf_array, crf_pad


def unpad_crf(returned_array, returned_mask, org_array, org_mask):
    out_array = org_array.clone().detach()
    out_array[org_mask] = returned_array[returned_mask]
    return out_array
