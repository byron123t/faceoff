import typing

from scaleatt.defenses.prevention.MedianFilteringDefense import MedianFilteringDefense
from scaleatt.defenses.prevention.RandomFilteringDefense import RandomFilteringDefense
from scaleatt.defenses.prevention.PreventionDefense import PreventionDefense
from scaleatt.defenses.prevention.PreventionDefenseType import PreventionTypeDefense

from scaleatt.scaling.ScalingApproach import ScalingApproach
from scaleatt.defenses.detection.fourier.FourierPeakMatrixCollector import FourierPeakMatrixCollector



class PreventionDefenseGenerator:
    """
    Generator for various prevention defenses.
    """

    @staticmethod
    def create_prevention_defense(defense_type: PreventionTypeDefense, verbose_flag: bool,
                                  scaler_approach: ScalingApproach,
                                  fourierpeakmatrixcollector: FourierPeakMatrixCollector,
                                  bandwidth: typing.Optional[int],
                                  usecythonifavailable: bool) -> PreventionDefense:
        """
        Creates a specific prevention defense.
        """

        if defense_type == PreventionTypeDefense.medianfiltering:
            preventiondefense: PreventionDefense = MedianFilteringDefense(verbose=verbose_flag,
                                                                                scaler_approach=scaler_approach,
                                                                                fourierpeakmatrixcollector=fourierpeakmatrixcollector,
                                                                                bandwidth=bandwidth, usecython=usecythonifavailable)
        elif defense_type == PreventionTypeDefense.randomfiltering:
            preventiondefense: PreventionDefense = RandomFilteringDefense(verbose=verbose_flag,
                                                                               scaler_approach=scaler_approach,
                                                                               fourierpeakmatrixcollector=fourierpeakmatrixcollector,
                                                                               bandwidth=bandwidth, usecython=usecythonifavailable)
        else:
            raise NotImplementedError()

        return preventiondefense

