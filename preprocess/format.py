#!/usr/bin/env python
from functools import partial

import click


@click.group()
def cli():
    pass


def extract_tag(line, coarse=False):
    line = line.strip()
    if line:
        word, tag = line.split()
        if coarse:
            tag = tag[0]
        return f'{word}\t{tag}\n'
    else:
        return '\n'


@cli.command()
@click.argument('input', type=click.File('r'))
@click.argument('output', type=click.File('w+'))
@click.option('--tag_type', default='fine',
              type=click.Choice(['coarse', 'fine'], case_sensitive=False))
def format_tags(input, output, tag_type):
    if tag_type == "coarse":
        f = partial(extract_tag, coarse=True)
    else:
        f = partial(extract_tag, coarse=False)
    for line in input:
        output.write(f(line))


@cli.command()
@click.argument('inputs', nargs=-1)
@click.argument('output', type=click.File('w+'))
def gather_tags(inputs, output):
    tags = set()
    for input in inputs:
        with open(input) as f:
            for line in f:
                line = line.strip()
                if line:
                    _, tag = line.split()
                    tags.add(tag)
    for tag in sorted(list(tags)):
        output.write(f'{tag}\n')


if __name__ == '__main__':
    cli()
