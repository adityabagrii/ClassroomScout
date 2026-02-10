import time
from typing import List, Optional, Callable

from .logging import log_msg
from .workflow import WFState


def run_once_for_course(app, course_id: str) -> dict:
    state = WFState(course_id=course_id)
    return app.invoke(state)


def run_once_all_courses_with_factory(
    courses: List[dict],
    app_factory: Callable[[dict], object],
    course_ids: Optional[List[str]] = None,
) -> None:
    if course_ids:
        ids = course_ids
        log_msg(f"courses.selected count={len(ids)}")
        selected = [c for c in courses if c.get("id") in ids]
    else:
        selected = courses

    for c in selected:
        cid = c.get("id")
        if not cid:
            continue
        log_msg(f"workflow.run course_id={cid}")
        try:
            app = app_factory(c)
            run_once_for_course(app, cid)
        except Exception as e:
            log_msg(f"workflow.error course_id={cid} error={e}")


def run_loop(
    courses: List[dict],
    app_factory: Callable[[dict], object],
    poll_hours: float,
    course_ids: Optional[List[str]] = None,
) -> None:
    sleep_s = max(60, int(poll_hours * 3600))
    log_msg(f"loop.start poll_hours={poll_hours} sleep_seconds={sleep_s}")
    while True:
        run_once_all_courses_with_factory(courses, app_factory, course_ids)
        log_msg(f"loop.sleep seconds={sleep_s}")
        time.sleep(sleep_s)
