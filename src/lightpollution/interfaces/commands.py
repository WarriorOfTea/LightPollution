"""Thin command handlers; all calculations use the application service."""

import json

from ..adapters.sources import LocalSourceRepository, RESOURCES
from ..adapters.storage import read_json, write_json
from ..application.planner import Planner, compare_results, export_result
from ..domain.scene import SceneSpec


def offline_example():
    return read_json(RESOURCES / 'example.json')


def run_command(args):
    if args.command == 'serve':
        import uvicorn
        from .web import create_app
        uvicorn.run(create_app(args.workspace, args.raw_directory), host=args.host, port=args.port)
    elif args.command == 'example':
        write_json(args.output, offline_example())
        print(f'Сценарий сохранён: {args.output}')
    elif args.command == 'datasets':
        repository = LocalSourceRepository(args.workspace, args.raw_directory)
        if args.download:
            repository.download(args.download)
        print(json.dumps(repository.catalogue(), ensure_ascii=False, indent=2))
    elif args.command == 'compare':
        value = compare_results(read_json(args.left), read_json(args.right))
        if args.output:
            write_json(args.output, value)
        print(json.dumps(value, ensure_ascii=False, indent=2))
    else:
        spec = SceneSpec.from_mapping(read_json(args.scene) if args.scene else offline_example())
        planner = Planner(args.workspace, raw_directory=args.raw_directory)
        result = planner.run(spec)
        export_result(result, args.output)
        value = result.value
        print(json.dumps({'output': str(args.output), 'directions': len(value['sky']['directions']),
            'stars': len(value['stars']['rows']), 'cache': value['cache'], 'runtime': value['runtime'],
            'physical_error_bound': None}, ensure_ascii=False, indent=2))
