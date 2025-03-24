import argparse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from parnassus.configs import Config
from parnassus.configs.pipeline import JetClusteringConfig
from parnassus.pipelines import JetClusteringPipeline, generate
from parnassus.utils.logger import setup_logger


def parse_args(args: Sequence[str] | None):
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("-c", "--config", type=str)
    _ = parser.add_argument("-i", "--input_path", type=str, default=None)
    _ = parser.add_argument("-o", "--output_path", type=str, default=None)
    _ = parser.add_argument("-n", "--num_steps", type=int, default=None)
    _ = parser.add_argument("-ne", "--num_events", type=int, default=None)
    _ = parser.add_argument("-bs", "--batch_size", type=int, default=None)
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    parsed_args = parse_args(args)
    log = setup_logger()
    title = " Starting Parnassus "
    log.info(f"[bold green]{title:-^100}")
    start = datetime.now()
    log.info(f"Start time: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    config = Config.from_yaml(parsed_args.config)

    if parsed_args.input_path:
        config.dataset_config.file_path = Path(parsed_args.input_path).absolute()
    if parsed_args.output_path:
        config.output_path = Path(parsed_args.output_path).absolute()
    if parsed_args.num_events:
        config.dataset_config.num_events = parsed_args.num_events
    if parsed_args.batch_size:
        config.batch_size = parsed_args.batch_size
    if parsed_args.num_steps:
        config.num_steps = parsed_args.num_steps

    gen_events = generate(config)
    log.info("[green]Starting postprocessing.")
    for pipeline_config in config.pipeline_configs:
        if isinstance(pipeline_config, JetClusteringConfig):
            pipeline = JetClusteringPipeline(pipeline_config)
            pipeline.process(gen_events)
    print(gen_events[0])
    end = datetime.now()
    title = " Completed! "
    log.info(f"[bold green]{title:-^100}")
    log.info(f"End time: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Elapsed time: {str(end - start).split('.')[0]}")


if __name__ == "__main__":
    main()
