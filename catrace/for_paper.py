import os
import json
import inspect
from os.path import join as pjoin


def save_stats_json(stats, stats_name, paper_fig_dir, tuple_key_to_str=False):
    if tuple_key_to_str:
        if 'raw' in stats.keys():
            stats['raw'] = {'_'.join(k): v for k, v in stats['raw'].items()}
            stats['shuffled'] = {'_'.join(k): v for k, v in stats['shuffled'].items()}
    stats_file = pjoin(paper_fig_dir, f'{stats_name}.json')
    with open(stats_file, 'w') as file:
        json.dump(stats, file)


def save_figure_for_paper(fig, fig_name, paper_fig_dir):
    fig.savefig(pjoin(paper_fig_dir, f'{fig_name}.svg'), transparent=True)
    fig.savefig(pjoin(paper_fig_dir, f'{fig_name}.pdf'), transparent=True)
    # Also save the current notebook path as a text file
    notebook_path = os.path.abspath('.')
    path_file = pjoin(paper_fig_dir, f'{fig_name}_notebook_path.txt')
    with open(path_file, 'w') as f:
        f.write(notebook_path)


def save_vsfigs_for_paper(output_figs, dataset_name, metric, paper_fig_dir, ytick_interval, adjust_vsfig):
    for vsname, vsfig in output_figs['vsfigs'].items():
        vsfig_copy = adjust_vsfig(vsfig, ytick_interval)
        vsfig_name = f'{dataset_name}_{metric}_{vsname}'
        save_figure_for_paper(vsfig_copy, vsfig_name, paper_fig_dir)