"""The persisted-lighting audit must detect the original empty-black-sky state."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch


class SavedLightingTest(unittest.TestCase):
    def test_missing_source_normals_must_have_saved_lod_recompute_flag(self):
        class Options:
            enabled=False
            def get_editor_property(self,name):
                self.assert_name=name
                return self.enabled
            def set_editor_property(self,*args): raise AssertionError("Readback must never mutate build settings")
        options=Options()
        mesh=types.SimpleNamespace(get_path_name=lambda:"/Game/Research/CityCodeOpenSource/Imported/Background/0_node_mesh.0_node_mesh")
        subsystem=types.SimpleNamespace(get_lod_build_settings=lambda mesh,index:options)
        spec=importlib.util.spec_from_file_location("ue_normal_readback_test",Path(__file__).with_name("ue_readback.py"))
        module=importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules",{"unreal":types.SimpleNamespace()}):spec.loader.exec_module(module)
        requirement={"recompute_normals_required":True,"missing_normal_primitives":[0]}
        actual,issues=module.inspect_generated_normal_requirement(mesh,requirement,subsystem)
        self.assertFalse(actual["actual_lod0_recompute_normals"])
        self.assertEqual(issues,["missing_source_normals_not_recomputed"])
        options.enabled=True
        actual,issues=module.inspect_generated_normal_requirement(mesh,requirement,subsystem)
        self.assertEqual(issues,[]);self.assertTrue(actual["actual_lod0_recompute_normals"])

    def test_reads_valid_reused_lights_and_rejects_missing_sky_and_black_hemisphere(self):
        class Component:
            def __init__(self, **values): self.values = values
            def get_editor_property(self, name): return self.values[name]
            def set_editor_property(self, *args): raise AssertionError("Readback must never mutate lighting")
        class Actor:
            def __init__(self, label, component=None): self.label, self.component = label, component
            def get_actor_label(self): return self.label
            def get_path_name(self): return "/SavedMap/"+self.label
            def get_component_by_class(self, cls): return self.component
        class Sun(Actor): pass
        class Sky(Actor): pass
        class Atmosphere(Actor): pass
        fake = types.SimpleNamespace(DirectionalLight=Sun, SkyLight=Sky, SkyAtmosphere=Atmosphere,
            DirectionalLightComponent=object, SkyLightComponent=object,
            ComponentMobility=types.SimpleNamespace(MOVABLE="movable"))
        spec = importlib.util.spec_from_file_location("ue_lighting_readback_test", Path(__file__).with_name("ue_readback.py"))
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"unreal":fake}): spec.loader.exec_module(module)
        sun = Sun("CityCode_InspectionSun", Component(mobility="movable", intensity=50000.,
            atmosphere_sun_light=True, atmosphere_sun_light_index=0))
        # The shared helper can reuse an existing, differently-labelled SkyLight.
        sky = Sky("SkyLight", Component(mobility="movable", intensity=1., real_time_capture=True,
            lower_hemisphere_is_black=False))
        environment, issues = module.inspect_saved_lighting([sun,sky,Atmosphere("Sky")])
        self.assertEqual(issues, [])
        self.assertTrue(environment["sky_atmosphere_present"])
        self.assertEqual(environment["sun"]["intensity"], 50000.)
        self.assertEqual(environment["skylight"]["actor_label"], "SkyLight")
        sky.component.values["lower_hemisphere_is_black"] = True
        environment, issues = module.inspect_saved_lighting([sun,sky])
        self.assertFalse(environment["sky_atmosphere_present"])
        self.assertIn("sky_atmosphere:missing_actor", issues)
        self.assertIn("skylight:lower_hemisphere_is_black_mismatch", issues)


if __name__ == "__main__": unittest.main()
