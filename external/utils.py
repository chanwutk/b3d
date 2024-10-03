import json
from typing import Any, Iterator


def jsonl_reader(filename: str) -> Iterator[tuple[int, Any]]:
    annotations_reader = open(filename, 'r')
    cache: dict[int, tuple[int, Any]] = {}
    idx = 0
    while True:
        if idx in cache:
            yield cache[idx]
            del cache[idx]
            idx += 1
        annotation_txt = annotations_reader.readline()
        try:
            annotations = json.loads(annotation_txt)
        except json.JSONDecodeError as e:
            if len(annotation_txt) == 0:
                break
            raise e

        assert isinstance(annotations, list)
        assert len(annotations) == 2
        _idx, _anns = annotations

        assert isinstance(_idx, int)
        anns = (_idx, _anns)

        if annotations[0] == idx:
            yield anns
            idx += 1
        else:
            cache[idx] = anns
    annotations_reader.close()
