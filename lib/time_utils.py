import time


class Est:
    def __init__(self, total):
        self.total = total
        self.ing = False
        self.timing = []
    
    def start(self):
        if self.ing:
            raise Exception('already timing')
        self.ing = True
        self.begin = time.time()

    def stop(self):
        if not self.ing:
            raise Exception('timer hasn\'t started')
        self.ing = False
        self.lastest_stop = time.time()
        self.timing.append(self.lastest_stop - self.begin)

    def est(self) -> str:
        count = len(self.timing)
        if count == 0:
            return 'inf'
        avg = 0
        for i in self.timing:
            avg += i
        avg /= count
        est = avg * (self.total - count) - time.time() + self.lastest_stop
        if est < 60:
            return f'{est:.2f} seconds'
        elif est < 3600:
            return f'{int(est / 60)} minutes and {int((est % 60))} seconds'
        elif est < 86400:
            return f'{int(est / 3600)} hours and {int((est % 3600) / 60)} minutes'
        else:
            return f'{int(est / 86400)} days'