"""An extra configuration file for pytest."""

num_top_durations: int = 10

def pytest_terminal_summary(terminalreporter, exitstatus):
    # Gather each report
    reports = []
    for replist in terminalreporter.stats.values():
        for rep in replist:
            if hasattr(rep, 'duration'):
                reports.append(rep)

    if not reports:
        return

    # Sort by execution time (duration) in ascending order
    reports.sort(
        key=lambda rep: rep.duration,
        reverse=True
    )
    reports = list(filter(
        lambda rep: (
            rep.when == "call" and
            rep.duration > 1  # milliseconds
        ),
        reports
    ))
    # Fetch the top 10 only
    reports = reports[:num_top_durations]

    terminalreporter.write_sep("=", f"Top {num_top_durations:d} call durations over 1s")
    for rep in reports:
        nodeid = rep.nodeid.replace("::()::", "::")
        terminalreporter.write_line("%02.2fs call %s" % (rep.duration, nodeid))
