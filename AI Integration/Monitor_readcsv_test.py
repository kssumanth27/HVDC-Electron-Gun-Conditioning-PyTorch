from datetime import datetime

from p4p.client.thread import Context


def make_callback(name: str):
    def callback(value):
        print(f"{name}: {value}")
        print("Updated:", datetime.fromtimestamp(value.timestamp))
        print("---")

    return callback


def main():
    ctxt = Context("pva")

    monitors = [
        ctxt.monitor("example:current", make_callback("current")),
        ctxt.monitor("example:pressure", make_callback("pressure")),
        ctxt.monitor("example:radiation", make_callback("radiation")),
        ctxt.monitor("example:voltage", make_callback("voltage")),
    ]

    input("Press Enter to quit...")


if __name__ == "__main__":
    main()