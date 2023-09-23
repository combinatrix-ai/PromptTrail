import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('error_filepath') 
    args = parser.parse_args()
    with open(args.error_filepath) as f:
        error_message = f.read()
    print('# Error Summarization')
    print('\n')
    print('TODO: impl summarization of error_message by PromptTail.\n')
    print(f'error_mesasge len: {len(error_message)}\n')
    print('\n')


if __name__ == "__main__":
    main()
