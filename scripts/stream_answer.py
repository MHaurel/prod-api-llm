import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream an answer from the API.")
    parser.add_argument("prompt", help="Prompt to send to the model")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/answers/stream",
        help="Streaming endpoint URL",
    )
    args = parser.parse_args()

    event = ""

    with httpx.stream(
        "POST",
        args.url,
        json={"prompt": args.prompt},
        timeout=None,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data = json.loads(line.removeprefix("data: "))

                if event == "token":
                    print(data["content"], end="", flush=True)
                elif event == "error":
                    raise RuntimeError(data["message"])
                elif event == "done":
                    print()


if __name__ == "__main__":
    main()
