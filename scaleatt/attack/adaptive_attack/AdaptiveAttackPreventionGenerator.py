import typing

from scaleatt.defenses.prevention.MedianFilteringDefense import MedianFilteringDefense
from scaleatt.defenses.prevention.RandomFilteringDefense import RandomFilteringDefense
from scaleatt.defenses.prevention.PreventionDefense import PreventionDefense
from scaleatt.defenses.prevention.PreventionDefenseType import PreventionTypeDefense

from scaleatt.scaling.ScalingApproach import ScalingApproach

from scaleatt.attack.adaptive_attack.AdaptiveAttack import AdaptiveAttackOnAttackImage
from scaleatt.attack.adaptive_attack.AdaptiveMedianAttack import AdaptiveMedianAttack
from scaleatt.attack.adaptive_attack.AdaptiveRandomAttack import AdaptiveRandomAttack


class AdaptiveAttackPreventionGenerator:
    """
    Generator for various adaptive attacks against prevention / filtering defenses.
    """

    @staticmethod
    def create_adaptive_attack(defense_type: PreventionTypeDefense, verbose_flag: bool,
                                  scaler_approach: ScalingApproach,
                                  preventiondefense: PreventionDefense,
                                  choose_only_unused_pixels_in_overlapping_case: bool,
                                  usecythonifavailable: bool,
                                  allowed_changes: typing.Optional[float]) -> AdaptiveAttackOnAttackImage:
        """
        Creates a specific adaptive attack against defense.
        """

        if defense_type == PreventionTypeDefense.medianfiltering:
            assert isinstance(preventiondefense, MedianFilteringDefense)
            adaptiveattack: AdaptiveMedianAttack = AdaptiveMedianAttack(verbose=verbose_flag,
                                                                              scaler_approach=scaler_approach,
                                                                              medianfilteringdefense=preventiondefense,
                                                                              choose_only_unused_pixels_in_overlapping_case=choose_only_unused_pixels_in_overlapping_case,
                                                                              allowed_ratio_of_change=allowed_changes,
                                                                              usecython=usecythonifavailable)

        elif defense_type == PreventionTypeDefense.randomfiltering:
            assert isinstance(preventiondefense, RandomFilteringDefense)
            adaptiveattack: AdaptiveRandomAttack = AdaptiveRandomAttack(verbose=verbose_flag,
                                                                              scaler_approach=scaler_approach,
                                                                              randomfilteringdefense=preventiondefense,
                                                                              choose_only_unused_pixels_in_overlapping_case=choose_only_unused_pixels_in_overlapping_case,
                                                                              allowed_ratio_of_change=allowed_changes,
                                                                              usecython=usecythonifavailable)
        else:
            raise NotImplementedError()

        return adaptiveattack

