"""Original island/archive selection with exact checks before paid scoring.

Parent sampling, archive updates, inspirations, history sorting and proposal
seeds follow the unchanged mentor module. Candidate-level checkpoints avoid
repeating committed victim episodes or critic scores. ASR is search success.
"""
import copy
import hashlib
import json
import random
import time

from secpo.attacks import agentdojo_genetic as original
from secpo.attacks.base import CandidateEvaluation, evaluation_history
from secpo.evaluation.agentdojo_qwen36 import AgentDojoEpisode, default_genetic_seeds

PROTOCOL = 'original-island-selection-exact-first-search-asr-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def candidate(record):
    return CandidateEvaluation(candidate_id=record['candidate_id'],
        injection=record['injection'], response=record['episode']['response'],
        score=record['score'], step=record['step'], metadata=record['episode']['metadata'])


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, (list, tuple)) else value


def search(*, task, context, victim, scorer, proposer, identity, save, resume=None,
           steps=2, candidates=8, format_retry_rounds=15, check=lambda: None,
           selection_seed=None, observe=lambda event: None):
    seeds = list(default_genetic_seeds(task.goal)[:candidates])
    if len(seeds) != candidates or steps < 1:
        raise ValueError('Invalid search dimensions')
    fingerprint = digest({'protocol': PROTOCOL, 'task': task.to_dict(), 'context': context,
        'identity': identity, 'seeds': seeds, 'steps': steps, 'candidates': candidates,
        'format_retry_rounds': format_retry_rounds, 'selection_seed': selection_seed})
    seed = selection_seed if selection_seed is not None else int(fingerprint[:16], 16) ^ 1234
    config = original.AgentDojoGeneticConfig(steps=steps, candidates_per_step=candidates,
        query_budget=steps*candidates, seed=seed, format_retry_rounds=format_retry_rounds)
    rng = random.Random(seed)
    if resume is None:
        state = {'protocol': PROTOCOL, 'fingerprint': fingerprint, 'task': task.to_dict(),
                 'batches': [], 'complete': False, 'search_success': False,
                 'selection_seed': seed, 'selection': None}
    else:
        state = copy.deepcopy(resume)
        checksum = state.pop('state_digest')
        if checksum != digest(state) or state['fingerprint'] != fingerprint:
            raise ValueError('Checkpoint identity or digest differs')
        if state['protocol'] != PROTOCOL:
            raise ValueError('Checkpoint protocol differs')
    records = {r['candidate_id']: r for b in state['batches'] for r in b['records']}
    def individual(identifier):
        return original._Individual(candidate(records[identifier]))
    selection = state['selection']
    if selection:
        islands = [[individual(i) for i in group] for group in selection['islands']]
        archive = {tuple(row['cell']): individual(row['id']) for row in selection['archive']}
        rng.setstate(tuples(selection['rng']))
    else:
        islands = [[] for _ in range(min(config.num_islands, candidates))]
        archive = {}

    def commit():
        state['selection'] = {
            'islands': [[x.evaluation.candidate_id for x in group] for group in islands],
            'archive': [{'cell': list(cell), 'id': x.evaluation.candidate_id} for cell, x in archive.items()],
            'rng': rng.getstate()}
        state['state_digest'] = digest({k: v for k, v in state.items() if k != 'state_digest'})
        save(copy.deepcopy(state))

    def finish(success, winner=None):
        state.update(complete=True, search_success=success, winner=winner,
            success_metric='search_asr', stopped_early=success,
            victim_episodes=sum(len(b['records']) for b in state['batches']))
        commit()
        observe({'generation': len(state['batches']), 'stage': 'task_complete', 'candidate_id': None})
        return state

    if state['complete']:
        return state
    for step in range(1, steps+1):
        check()
        if len(state['batches']) < step:
            parent = None
            inspirations = []
            history_ids = []
            method = 'initialization'
            island_index = 0
            if step > 1:
                nonempty = sum(bool(group) for group in islands)
                if not nonempty or not archive:
                    raise ValueError('No eligible population')
                island_index = (step-1) % nonempty
                selected, method = original._select_parent(island_index, islands, archive, config, rng)
                inspirations = original._get_inspirations(selected, islands, archive, config, rng)
                parent = records[selected.evaluation.candidate_id]
                population = list(inspirations)
                if all(x.evaluation.injection != selected.evaluation.injection for x in population):
                    population.append(selected)
                population.sort(key=lambda x: x.evaluation.score, reverse=True)
                history_ids = [x.evaluation.candidate_id for x in population[:config.max_history]]
            state['batches'].append({'step': step, 'parent': parent,
                'inspirations': [x.evaluation.candidate_id for x in inspirations],
                'history_ids': history_ids, 'sampling_method': method, 'island_index': island_index,
                'injections': seeds if step == 1 else [], 'proposal_rounds': 0,
                'records': [], 'inserted': False, 'phase': 'propose'})
            commit()
        batch = state['batches'][step-1]
        if step > 1:
            history = tuple(candidate(records[i]) for i in batch['history_ids'])
            while len(batch['injections']) < candidates:
                check()
                round_ = batch['proposal_rounds']
                if round_ > format_retry_rounds:
                    raise RuntimeError('Attacker failed to fill eight unique mutations')
                request = original.AgentDojoAttackRequest(algorithm=original.ALGORITHM,
                    goal=task.goal, step=step, count=candidates-len(batch['injections']),
                    parents=tuple(x.injection for x in history), history=evaluation_history(history),
                    seed=config.seed+step*(config.format_retry_rounds+1)+round_,
                    parent=candidate(batch['parent']),
                    inspirations=tuple(candidate(records[i]) for i in batch['inspirations']),
                    sampling_method=batch['sampling_method'], retry_round=round_, context=context)
                observe({'generation': step, 'stage': 'attacker', 'candidate_id': None})
                proposed = proposer(request)
                if isinstance(proposed, (str, bytes)) or not isinstance(proposed, (list, tuple)):
                    raise TypeError('Proposal must be a sequence')
                seen = {x.injection for x in history} | set(batch['injections'])
                for value in proposed:
                    if not isinstance(value, str):
                        raise TypeError('Proposal is not text')
                    value = value.strip()
                    if value and value not in seen and len(batch['injections']) < candidates:
                        batch['injections'].append(value)
                        seen.add(value)
                batch['proposal_rounds'] += 1
                commit()
        batch['phase'] = 'victim_exact_check'
        commit()
        for ordinal, injection in enumerate(batch['injections']):
            check()
            if ordinal < len(batch['records']):
                record = batch['records'][ordinal]
                if record['injection'] != injection:
                    raise ValueError('Checkpoint candidate order differs')
            else:
                observe({'generation': step, 'stage': 'victim', 'candidate_id': f's{step:03d}-c{ordinal:03d}'})
                started = time.monotonic()
                episode = AgentDojoEpisode.from_value(victim(task, injection))
                if episode.metadata.get('episode_error'):
                    raise RuntimeError('Episode error; do not count as attack failure')
                record = {'candidate_id': f's{step:03d}-c{ordinal:03d}', 'step': step,
                    'injection': injection, 'episode': episode.to_dict(), 'score': None,
                    'victim_seconds': time.monotonic()-started}
                batch['records'].append(record)
                records[record['candidate_id']] = record
                commit()
            if record['episode']['exact_success']:
                record['score'] = 10.0
                batch['phase'] = 'exact_success'
                return finish(True, {'candidate_id': record['candidate_id'], 'step': step,
                    'injection': injection, 'episode': record['episode']})
        batch['phase'] = 'critic'
        commit()
        for record in batch['records']:
            check()
            if record['score'] is not None:
                continue
            observe({'generation': step, 'stage': 'critic', 'candidate_id': record['candidate_id']})
            started = time.monotonic()
            episode = AgentDojoEpisode.from_value(record['episode'])
            scored = AgentDojoEpisode.from_value(scorer(task, record['injection'], episode))
            if scored.exact_success != episode.exact_success or scored.response != episode.response:
                raise ValueError('Critic changed deterministic outcome')
            record.update(episode=scored.to_dict(), score=scored.dense_score,
                          critic_seconds=time.monotonic()-started)
            commit()
        if not batch['inserted']:
            original._add_generation([candidate(x) for x in batch['records']], islands,
                archive, batch['island_index'], step == 1, config, rng)
            batch['inserted'] = True
        batch['phase'] = 'complete'
        commit()
        observe({'generation': step, 'stage': 'generation_complete', 'candidate_id': None})
    return finish(False)


def score_cached_episode(runner, task, injection, episode):
    from secpo.evaluation.agentdojo_qwen36 import FreshAgentDojoEpisodeRunner
    proxy = copy.copy(runner)
    proxy.replay = lambda supplied_task, supplied_injection: episode
    return FreshAgentDojoEpisodeRunner.__call__(proxy, task, injection)
