from context.validation.correctness.train_comparison import linux_rss


def test_linux_rss_counts_descendants_once_and_tolerates_exited_workers(tmp_path):
    for pid, rss, children in ((1, 100, '2 3'), (2, 200, '4'), (4, 300, '')):
        root = tmp_path / str(pid)
        (root / 'task' / str(pid)).mkdir(parents=True)
        (root / 'status').write_text(f'Name:\tworker\nVmRSS:\t{rss} kB\n')
        (root / 'task' / str(pid) / 'children').write_text(children)
    # A second parent thread reports the same child; PID 3 has already exited.
    (tmp_path / '1/task/11').mkdir()
    (tmp_path / '1/task/11/children').write_text('2')
    assert linux_rss(1, tmp_path) == dict(rss=100*1024, worker_rss=500*1024)
