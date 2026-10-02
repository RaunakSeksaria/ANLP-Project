#!/usr/bin/env python3
"""Read-only audit of source story memberships and saved positive pairs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
# These judgments follow manual review of both full original bodies, not a text classifier.
# Keys are indices in the saved sampled-triplet CSV; assertions below lock the seed-42 draw.
REVIEW = {
    170: ('related_developments', 'Same Parkland aftermath, but distinct Trump gun-policy responses: assault-weapon ban versus arming teachers.'),
    176: ('same_specific_story', "John Kerry's tearful public update on Teresa Heinz Kerry's hospitalization."),
    187: ('different_events', 'Pompeo interview about a Trump–Taliban meeting versus comments about a possible Trump–Rouhani meeting with Iran.'),
    255: ('same_specific_story', "Richard M. Daley's hospitalization and ICU stay."),
    400: ('same_specific_story', "Obama's decision not to re-nominate Michael Boggs to the federal bench."),
    859: ('different_events', 'George W. Bush on Jimmy Kimmel versus a separate appearance on Ellen about his inauguration poncho.'),
    870: ('related_developments', 'Trump–Abe visit links the articles, but one reports an immigration-order press conference and the other a later golf/press-pool incident.'),
    900: ('same_specific_story', "Obama's Chicago farewell address; one report and one response letter."),
    1024: ('same_specific_story', 'Different answers in the same Obama–Diane Sawyer interview after the Romney debate.'),
    1048: ('related_developments', 'Trump inauguration boycott context, but a broad map of boycotting House Democrats versus Trump reacting specifically to John Lewis.'),
    1298: ('same_specific_story', "Pence's Israel visit, including the same Tuesday Western Wall visit; one straight report and one retrospective."),
    1386: ('related_developments', 'Same E. Jean Carroll allegation, but a reaction to a specific Trump remark versus a broader opinion letter about testimony.'),
    1431: ('same_specific_story', 'The shooting at Marjory Stoneman Douglas High School in Parkland.'),
    1466: ('different_events', 'Separate Sean Spicer book-tour incidents: a Rhode Island racial-slur accusation versus heckling in New York City.'),
    1517: ('same_specific_story', "Margaret Thatcher's death and reactions to it, including a U.S. House tribute."),
    1533: ('related_developments', "The 'pencil-neck Adam Schiff' insult connects them; merchandise announcement versus a broader opinion piece on Trump's insults."),
    1568: ('topic_only_or_insufficient', "Herman Cain's 9-9-9 plan appears as article subject versus only in links to debate clips under a Romney headline; no common occurrence is established."),
    1678: ('same_specific_story', "The proposed Iraqi insurgent amnesty plan and the same controversy over whether attackers of U.S. soldiers would be pardoned."),
    1705: ('same_specific_story', "Danny Glover/MoveOn push to nominate Paul Krugman for Treasury secretary; the left article describes a response petition."),
    1943: ('same_specific_story', "The Comey memo allegation that Trump asked him to end the Michael Flynn investigation."),
}
EXPECTED_SELECTED = sorted(REVIEW)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def norm_text(value: str, limit: int = 230) -> str:
    text = ' '.join(value.split())
    return text[:limit] + ('…' if len(text) > limit else '')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=ROOT.parent/'embedding_data/metadata.parquet')
    parser.add_argument('--triplets', type=Path, default=ROOT/'results/sampled_triplets.csv')
    parser.add_argument('--source', type=Path, default=ROOT.parent/'dataset/raw/neutral_articles_10k.jsonl')
    parser.add_argument('--output-prefix', type=Path, default=ROOT/'results/event_id_audit')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    metadata = pq.read_table(args.metadata, columns=['sample_id', 'ideology', 'event_ids', 'story_id', 'title', 'original_article', 'original_article_sha256']).to_pylist()
    by_sample = {row['sample_id']: row for row in metadata}
    assert len(by_sample) == len(metadata)
    source_by_body = {(r['ideology'], r['original_article_sha256']): r for r in metadata}
    assert len(source_by_body) == len(metadata), 'Duplicate body hashes prevent exact source join'
    members_by_event: dict[str, list[str]] = defaultdict(list)
    for row in metadata:
        assert isinstance(row['event_ids'], list) and row['event_ids']
        assert all(isinstance(x, str) and x for x in row['event_ids'])
        assert len(row['event_ids']) == len(set(row['event_ids']))
        assert row['story_id'] in row['event_ids']
        for e in row['event_ids']:
            members_by_event[e].append(row['sample_id'])
    found_memberships: dict[str, set[str]] = defaultdict(set)
    source_ids: set[str] = set()
    title_variants = 0
    with args.source.open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            record = json.loads(line)
            event = str(record['story_id'])
            assert event not in source_ids, f'Duplicate source story_id {event}'
            source_ids.add(event)
            for side in ('left', 'right'):
                original = record[side]
                body_hash = hashlib.sha256(original['body'].encode('utf-8')).hexdigest()
                key = side, body_hash
                assert key in source_by_body, f'Unmatched {side} article in source row {line_number}'
                article = source_by_body[key]
                found_memberships[article['sample_id']].add(event)
                title_variants += original['title'] != article['title']
    assert source_ids == set(members_by_event)
    assert all(found_memberships[r['sample_id']] == set(r['event_ids']) for r in metadata)
    assert len(source_ids) == 10000 and len(metadata) == 11855
    assert all(len(v) == 2 and {by_sample[s]['ideology'] for s in v} == {'left','right'}
               for v in members_by_event.values())
    article_sizes = [len(r['event_ids']) for r in metadata]
    event_sizes = [len(v) for v in members_by_event.values()]
    distinct_pairings = Counter(tuple(sorted(v)) for v in members_by_event.values())
    with args.triplets.open(newline='') as f:
        triplets = list(csv.DictReader(f))
    selected = sorted(np.random.default_rng(args.seed).choice(len(triplets), 20, replace=False).tolist())
    assert selected == EXPECTED_SELECTED, 'Sampling changed; manual reviews require reinspection'
    reviews = []
    for idx in selected:
        triplet = triplets[idx]
        a, p = by_sample[triplet['anchor_id']], by_sample[triplet['positive_id']]
        common = sorted(set(a['event_ids']) & set(p['event_ids']))
        assert common and a['ideology'] != p['ideology']
        assert common == json.loads(triplet['shared_positive_events'])
        label, reason = REVIEW[idx]
        reviews.append({'triplet_id': triplet['triplet_id'], 'saved_triplet_index': idx,
            'shared_event_ids': common, 'judgment': label, 'reason': reason,
            'anchor': {'sample_id': a['sample_id'], 'ideology': a['ideology'], 'title': a['title'],
                       'article_beginning': norm_text(a['original_article'])},
            'positive': {'sample_id': p['sample_id'], 'ideology': p['ideology'], 'title': p['title'],
                         'article_beginning': norm_text(p['original_article'])}})
    summary = {
        'semantics': ('event_ids is a list of raw source story_id values for every left/right pairing in which '
                      'the exact original article body occurs. It is multi-valued and does not establish a '
                      'semantically exclusive or validated real-world event cluster.'),
        'datatype': 'Arrow list<string>; Python list[str]',
        'articles': len(metadata), 'source_pairing_rows': len(source_ids),
        'article_event_memberships': sum(article_sizes),
        'articles_with_multiple_event_ids': sum(n > 1 for n in article_sizes),
        'ids_per_article': {'min': min(article_sizes), 'median': statistics.median(article_sizes),
                            'mean': statistics.mean(article_sizes), 'max': max(article_sizes),
                            'histogram': dict(sorted(Counter(article_sizes).items()))},
        'articles_per_event_id': {'min': min(event_sizes), 'median': statistics.median(event_sizes),
                                  'mean': statistics.mean(event_sizes), 'max': max(event_sizes),
                                  'histogram': dict(sorted(Counter(event_sizes).items()))},
        'event_ids_with_both_ideologies': sum({by_sample[s]['ideology'] for s in v} == {'left','right'}
                                             for v in members_by_event.values()),
        'distinct_left_right_article_pairings': len(distinct_pairings),
        'source_ids_per_distinct_article_pair_histogram': dict(sorted(Counter(distinct_pairings.values()).items())),
        'source_membership_match': '20,000 of 20,000 exact-body/source-story memberships match metadata',
        'source_title_variants': title_variants,
        'review_sample': {'seed': args.seed, 'sampled_from_saved_triplets': len(triplets),
                          'count': len(reviews), 'judgment_counts': dict(Counter(r['judgment'] for r in reviews)),
                          'scope': ('A manual qualitative check of these 20 full original articles, not a '
                                    'validated precision estimate for all sampled triplets')},
        'inputs_sha256': {str(path): sha256(path) for path in (args.metadata, args.triplets, args.source)},
    }
    output = {'summary': summary, 'reviews': reviews}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix('.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# Audit of `event_ids` in saved positive pairs', '',
        'This file only inspects metadata, the original articles in the source pairing file, and saved triplet IDs. '
        'It does not recompute embeddings or triplet metrics.', '',
        f"**Memberships:** {summary['article_event_memberships']:,} across {summary['articles']:,} articles and "
        f"{summary['source_pairing_rows']:,} source story IDs. "
        f"{summary['articles_with_multiple_event_ids']:,} articles have multiple IDs.", '',
        f"**IDs per article:** min {summary['ids_per_article']['min']}, median {summary['ids_per_article']['median']}, "
        f"mean {summary['ids_per_article']['mean']:.4f}, max {summary['ids_per_article']['max']}.", '',
        f"**Articles per ID:** min/median/mean/max = 2/2/2/2. "
        f"All {summary['event_ids_with_both_ideologies']:,} IDs have one left and one right article.", '',
        f"The IDs represent {summary['distinct_left_right_article_pairings']:,} distinct article pairings; "
        'repeated pairings can carry several source story IDs.', '',
        f"**Source trace:** {summary['source_membership_match']}. Two occurrences have a title variant "
        'with the same exact article body; titles are not the membership key.', '',
        '## Manual interpretation', '',
        'An ID certifies a pairing in the source file. It does not certify that both stories cover one ' 
        'narrowly defined news event. Judgments here are conservative: `same_specific_story` includes '
        'reports and reactions to one identifiable occurrence or announcement; `related_developments` '
        'covers distinct developments in a common story arc; `different_events` names separate '
        'occurrences; `topic_only_or_insufficient` lacks enough evidence of a common event.', '',
        f"Judgments in the seed-{args.seed} sample: " + ', '.join(f'{k}: {v}' for k,v in summary['review_sample']['judgment_counts'].items()) + '.', '',
    ]
    for review in reviews:
        a, p = review['anchor'], review['positive']
        lines.extend([
            f"### {review['triplet_id']} — {review['judgment']}", '',
            f"Shared source ID(s): {', '.join(review['shared_event_ids'])}.", '',
            f"- Anchor **{a['sample_id']} ({a['ideology']})** — {a['title']}\n  - Beginning: {a['article_beginning']}",
            f"- Positive **{p['sample_id']} ({p['ideology']})** — {p['title']}\n  - Beginning: {p['article_beginning']}", '',
            f"Assessment: {review['reason']}", '',
        ])
    lines.extend(['## Implication', '',
        'The earlier count of “10,000 events containing both ideologies” describes 10,000 **source pairing IDs**. '
        'It overstates the number of independently verified real-world events. The saved triplet '
        'evaluation used IDs as positives; its numbers are mathematically reproducible, while '
        'the semantic positive labels contain observable noise. No triplet scores were changed by this audit.', ''])
    args.output_prefix.with_suffix('.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f"20 pair reviews: {args.output_prefix.with_suffix('.md')}")


if __name__ == '__main__':
    main()
