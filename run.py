import argparse

from researchloop.server import serve

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ResearchLoop local research workbench")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    serve(port=args.port)
