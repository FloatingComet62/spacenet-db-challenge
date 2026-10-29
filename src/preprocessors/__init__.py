PREPROCESSORS = {}

from preprocessors.bilateral import process as bilateral
PREPROCESSORS["bilateral"] = bilateral

from preprocessors.clahe import process as clahe
PREPROCESSORS["clahe"] = clahe

from preprocessors.gamma import process as gamma
PREPROCESSORS["gamma"] = gamma

from preprocessors.noop import process as noop
PREPROCESSORS["noop"] = noop

from preprocessors.sharpen import process as sharpen
PREPROCESSORS["sharpen"] = sharpen

PREPROCESSORS["chainA"] = lambda x: sharpen(clahe(x))
PREPROCESSORS["chainB"] = lambda x: clahe(bilateral(x))
PREPROCESSORS["chainC"] = lambda x: sharpen(clahe(bilateral(x)))
