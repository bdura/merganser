import duckietown_utils as dtu


class NoHomographyInfoAvailable(dtu.DTException):
    pass


class InvalidHomographyInfo(dtu.DTException):
    pass


class CouldNotCalibrate(dtu.DTException):
    pass
