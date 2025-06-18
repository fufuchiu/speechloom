"""Inspect local conversation data and multimodal token sequences."""

import argparse
import json

from .conversation import load_conversations
from .tokens import TokenLayout, encode_text, pack_response


def main(argv=None) -> int:
    """Print JSON summaries without downloading weights or accessing a network."""
    parser = argparse.ArgumentParser(prog='speechloom')
    commands = parser.add_subparsers(dest='command', required=True)
    token = commands.add_parser('tokenize')
    token.add_argument('text')
    token.add_argument('--audio-codes', default='')
    inspect = commands.add_parser('audit')
    inspect.add_argument('manifest')
    args = parser.parse_args(argv)
    try:
        if args.command == 'tokenize':
            ids = (
                pack_response(
                    args.text, [int(v) for v in args.audio_codes.split(',')], TokenLayout()
                )
                if args.audio_codes
                else encode_text(args.text, True)
            )
            result = {'ids': ids, 'tokens': len(ids)}
        else:
            conversations = load_conversations(args.manifest)
            result = {'conversations': len(conversations), 'turns': sum(map(len, conversations))}
    except (ValueError, OSError) as exc:
        parser.exit(2, f'error: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0
