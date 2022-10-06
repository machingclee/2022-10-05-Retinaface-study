from pydash.objects import get, set_


class ConsoleLog():
    def __init__(self, lines_up_on_end=0):
        self.CLR = "\x1B[0K"
        self.lines_up_on_batch_end = lines_up_on_end
        self.record = {}

    def UP(self, lines):
        return "\x1B[" + str(lines + 1) + "A"

    def DOWN(self, lines):
        return "\x1B[" + str(lines) + "B"

    def on_print_end(self):
        print(self.UP(self.lines_of_log))
        print(self.UP(self.lines_up_on_batch_end))

    def print(self, key_values):
        lines_of_log = len(key_values)
        self.lines_of_log = lines_of_log

        # for the first time,
        # print self.lines_of_log number of lines to occupy the space
        print("".join(["\n"] * (self.lines_of_log)))
        print(self.UP(self.lines_of_log))

        for key, value in key_values:
            if key == "" and value == "":
                print()
            else:
                if key != "" and value != "":
                    prev_value = get(self.record, key, 0.)
                    curr_value = value
                    diff = curr_value - prev_value
                    sign = "+" if diff >= 0 else ""
                    print("{0: <35} {1: <30}".format(key, value) + sign + "{:.5f}".format(diff) + self.CLR)
                    set_(self.record, key, value)

        self.on_print_end()

    def clear_log_on_epoch_end(self):
        # usually before calling this line, print() has been run, therefore we are at the top of the log.
        for _ in range(self.lines_of_log):
            # clear lines
            print(self.CLR)
        # ready for next epoch
        print(self.UP(self.lines_of_log))
