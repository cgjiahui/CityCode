"""Transient offscreen visual inspection inside UE Editor, never saves a map.

Set UE_CITYCODE_RENDER_JOB to a JSON job and start with -RenderOffscreen (without
-NullRHI) and -ExecCmds="py /absolute/path/ue_render.py". A Slate post-tick callback
keeps rendered frames progressing before export. Result success requires a real
PNG file, not merely a successful screenshot API call. UE 5.0 SceneCapture2D
does not reproduce the main viewport Lumen lighting; use base_color for remote
material/geometry inspection and mode=viewport for interactive lighting review.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import struct
import time
import traceback
import unreal

ROOT='/Game/Research/CityCodeOpenSource/Maps/'
STATE={}


def persist():
    destination=Path(STATE['job']['output_json']);destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(STATE['report'],indent=2,allow_nan=False)+'\n',encoding='utf-8')


def vector(value):return [float(value.x),float(value.y),float(value.z)]


def finish(error=None):
    if STATE.get('finished'):return
    STATE['finished']=True
    report=STATE['report']
    if error:report.update(status='failed',error=str(error),traceback=traceback.format_exc())
    else:report['status']='rendered_pending_visual_review'
    report['elapsed_seconds']=time.monotonic()-STATE['started']
    report['maps_saved']=False
    persist()
    handle=STATE.get('handle')
    if handle is not None:unreal.unregister_slate_post_tick_callback(handle)
    if STATE.get('component'):STATE['component'].set_editor_property('texture_target',None)
    if STATE.get('target'):unreal.RenderingLibrary.release_render_target2d(STATE['target'])
    if STATE.get('actor'):unreal.EditorLevelLibrary.destroy_actor(STATE['actor'])
    if STATE['job'].get('quit_when_done',True):unreal.SystemLibrary.quit_editor()


def bounds():
    boxes=[]
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        if not any(str(tag).startswith('CityCodeInput=')for tag in actor.get_editor_property('tags')):continue
        component=actor.get_component_by_class(unreal.StaticMeshComponent)
        if component is None:continue
        origin,extent=actor.get_actor_bounds(False,False);o,e=vector(origin),vector(extent)
        boxes.append([[o[i]/100-e[i]/100 for i in range(3)],[o[i]/100+e[i]/100 for i in range(3)]])
    if not boxes:raise ValueError('No tagged CityCode geometry exists in this map')
    return [[min(box[0][i]for box in boxes)for i in range(3)],[max(box[1][i]for box in boxes)for i in range(3)]]


def default_views(box):
    center=[(a+b)*.5 for a,b in zip(*box)];radius=max(b-a for a,b in zip(*box))
    distance=max(40.,radius*1.4)
    return [{'name':'southwest','location_m':[center[0]-distance,center[1]-distance,center[2]+distance*.8],
             'target_m':center,'fov_degrees':45.},
            {'name':'northeast','location_m':[center[0]+distance,center[1]+distance,center[2]+distance*.65],
             'target_m':center,'fov_degrees':45.}]


def setup_view(index):
    view=STATE['views'][index];location=view['location_m'];target=view['target_m']
    delta=[b-a for a,b in zip(location,target)]
    rotation=unreal.Rotator(pitch=math.degrees(math.atan2(delta[2],math.hypot(*delta[:2]))),
                            yaw=math.degrees(math.atan2(delta[1],delta[0])),roll=0.)
    actor=STATE['actor'];actor.set_actor_location_and_rotation(unreal.Vector(*(v*100 for v in location)),rotation,False,True)
    component=STATE['component'];component.set_editor_property('fov_angle',float(view.get('fov_degrees',45.)))
    mode=view.get('capture_mode',STATE['job'].get('capture_mode','lit'))
    if mode=='base_color':
        if not hasattr(unreal.SceneCaptureSource,'SCS_BASE_COLOR'):raise ValueError('This UE lacks deferred base-color capture')
        component.set_editor_property('capture_source',unreal.SceneCaptureSource.SCS_BASE_COLOR)
    else:component.set_editor_property('capture_source',unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR)
    if STATE['job'].get('inspection_daylight',False):
        STATE['report']['capture_exposure']=capture_daylight_exposure(component)
    component.set_editor_property('capture_every_frame',True)
    STATE.update(view_index=index,view_frame=0,stable_frames=0,view_started=time.monotonic(),
                 loading_barrier_completed=False,loading_barrier_attempted=False)
    STATE['report']['active_view']=view['name'];persist()


def compiling():
    for name in ('are_any_shader_compilations_running','are_any_shaders_compiling'):
        function=getattr(getattr(unreal,'AutomationLibrary',None),name,None)
        if function:
            try:return bool(function())
            except Exception:pass
    return None


def inspection_daylight():
    """Set up this Editor session, without saving assets or the assembled map.

    A captured SkyLight cannot light an empty black sky. The assembly's two
    default lights are reused, then a physical sky is supplied. The project's
    exposure is preserved. Re-running this script does not add duplicate lights.
    """
    actors=list(unreal.EditorLevelLibrary.get_all_level_actors())
    created=[]

    def find_or_spawn(actor_class,label,reuse_class=False):
        actor=next((item for item in actors if item.get_actor_label()==label),None)
        if actor is None and reuse_class:
            actor=next((item for item in actors if isinstance(item,actor_class)),None)
        if actor is None:
            actor=unreal.EditorLevelLibrary.spawn_actor_from_class(actor_class,unreal.Vector(0,0,0))
            actor.set_actor_label(label)
            actor.set_folder_path('CityCode/InspectionEnvironment')
            actors.append(actor);created.append(label)
        return actor

    sun=find_or_spawn(unreal.DirectionalLight,'CityCode_InspectionSun',True)
    sun.set_actor_rotation(unreal.Rotator(pitch=-48.,yaw=-35.,roll=0.),False)
    sun_component=sun.get_component_by_class(unreal.DirectionalLightComponent)
    sun_component.set_mobility(unreal.ComponentMobility.MOVABLE)
    sun_component.set_intensity(50000.)
    sun_component.set_editor_property('atmosphere_sun_light',True)
    sun_component.set_editor_property('atmosphere_sun_light_index',0)
    sky_atmosphere=find_or_spawn(unreal.SkyAtmosphere,'CityCode_InspectionAtmosphere',True)
    sky_atmosphere.set_actor_location(unreal.Vector(0,0,0),False,True)
    skylight=find_or_spawn(unreal.SkyLight,'CityCode_InspectionSkyLight',True)
    sky_component=skylight.get_component_by_class(unreal.SkyLightComponent)
    sky_component.set_mobility(unreal.ComponentMobility.MOVABLE)
    sky_component.set_intensity(1.)
    sky_component.set_editor_property('real_time_capture',True)
    sky_component.set_editor_property('lower_hemisphere_is_black',False)
    return {'preset':'inspection_daylight','session_only':True,'maps_saved':False,
            'created_actors':created,'sun_illuminance_lux':50000.,
            'sun_rotation_degrees':{'pitch':-48.,'yaw':-35.,'roll':0.},
            'skylight_intensity':1.,'skylight_realtime_capture':True,
            'exposure_method':'project_and_viewport_settings_preserved',
            'brightness_visual_validation':'pending_actual_image_review',
            'visual_readiness':'awaiting_shader_texture_mesh_preparation_and_image_review'}


def finish_loading_for_capture():
    """An offscreen-only barrier; the interactive viewport never calls this."""
    function=getattr(getattr(unreal,'AutomationLibrary',None),'finish_loading_before_screenshot',None)
    report=STATE['report']
    report['loading_barrier']={'available':callable(function),'completed':False}
    persist()
    if not callable(function):
        if compiling() is None:
            raise RuntimeError('Cannot establish capture readiness: shader query and loading barrier are both unavailable')
        report['loading_barrier']['shader_query_fallback']=True
        persist();return
    started=time.monotonic()
    function()
    STATE['loading_barrier_completed']=True
    report['loading_barrier'].update(completed=True,duration_seconds=time.monotonic()-started)
    # Let rendering catch up after the loading flush, rather than exporting the
    # frame that was assembled before shader/texture/mesh compilation finished.
    STATE.update(view_frame=0,stable_frames=0,view_started=time.monotonic())
    persist()


def capture_daylight_exposure(component):
    """Expose the diagnostic camera, without changing the Editor's exposure.

    The first actual 50,000-lux capture clipped every pixel to 255 despite a
    completed loading barrier. Give SceneCapture its own physical daylight
    camera instead of assuming it shares the interactive viewport adaptation.
    """
    settings=component.get_editor_property('post_process_settings')
    values={'auto_exposure_method':unreal.AutoExposureMethod.AEM_MANUAL,
            'auto_exposure_apply_physical_camera_exposure':True,
            'auto_exposure_bias':0.,'camera_iso':100.,
            'camera_shutter_speed':125.,'depth_of_field_fstop':11.,
            'depth_of_field_scale':0.}
    for name,value in values.items():
        settings.set_editor_property('override_'+name,True)
        settings.set_editor_property(name,value)
    component.set_editor_property('post_process_settings',settings)
    component.set_editor_property('post_process_blend_weight',1.)
    return {'scope':'SceneCapture component only; Editor and saved map exposure unchanged',
            'method':'manual_physical_camera','iso':100.,'shutter_inverse_seconds':125.,
            'aperture_fstop':11.,'ev100':math.log2(11.**2*125.),'exposure_bias':0.,
            'base_color_mode':'raw material output; exposure does not light this pass'}


def inspect_png(path):
    data=path.read_bytes()
    if len(data)<100 or data[:8]!=b'\x89PNG\r\n\x1a\n':raise ValueError('Render target export is absent or is not a PNG')
    width,height=struct.unpack('>II',data[16:24])
    return {'file':str(path),'bytes':len(data),'width':width,'height':height,'png_signature_valid':True}


def tick(delta_seconds):
    if STATE.get('finished'):return
    try:
        STATE['view_frame']+=1
        shader=compiling()
        # None means unknown on UE 5.0, not "no shader work". Only a completed
        # loading barrier can establish readiness when no query API is exposed.
        known_idle=shader is False or (shader is None and STATE.get('loading_barrier_completed',False))
        STATE['stable_frames']=STATE['stable_frames']+1 if known_idle else 0
        elapsed=time.monotonic()-STATE['view_started']
        job=STATE['job'];report=STATE['report']
        report['shader_query_available']=shader is not None
        if elapsed>float(job.get('max_view_seconds',480.)):
            raise RuntimeError('Visual QA did not finish within its finite render-task budget')
        if STATE['view_frame']<int(job.get('warmup_frames',90)) or elapsed<float(job.get('minimum_warmup_seconds',10.)):return
        if not STATE.get('loading_barrier_attempted',False):
            STATE['loading_barrier_attempted']=True
            finish_loading_for_capture()
            return
        if STATE['stable_frames']<int(job.get('shader_stable_frames',10)):return
        view=STATE['views'][STATE['view_index']]
        component=STATE['component'];component.set_editor_property('capture_every_frame',False)
        component.capture_scene()
        directory=Path(job['output_directory']);directory.mkdir(parents=True,exist_ok=True)
        filename=view['name']+'.png';path=directory/filename
        if path.exists():path.unlink()
        unreal.RenderingLibrary.export_render_target(STATE['world'],STATE['target'],str(directory),filename)
        result=inspect_png(path);result.update(view=view,warmup_frames=STATE['view_frame'],warmup_seconds=elapsed,
                                              shader_compilation_query=shader,visual_review='pending')
        result['loading_barrier']=dict(report['loading_barrier'])
        if report.get('capture_exposure'):result['capture_exposure']=dict(report['capture_exposure'])
        report['captures'].append(result);persist()
        next_index=STATE['view_index']+1
        if next_index<len(STATE['views']):setup_view(next_index)
        else:finish()
    except Exception as error:
        unreal.log_error('CityCode visual QA: '+str(error));finish(error)


def main():
    job_path=os.environ.get('UE_CITYCODE_RENDER_JOB')
    if not job_path:raise RuntimeError('Set UE_CITYCODE_RENDER_JOB to an absolute render job JSON')
    job=json.loads(Path(job_path).read_text(encoding='utf-8-sig'))
    level=job.get('map')
    if job.get('assembly_result'):
        assembled=json.loads(Path(job['assembly_result']).read_text(encoding='utf-8-sig'))
        if not assembled.get('ue_executed'):raise ValueError('Assembly result was not executed in UE')
        if level and assembled['new_level_path']!=level:raise ValueError('Render level differs from assembly result')
        level=assembled['new_level_path']
    if not level or not level.startswith(ROOT):raise ValueError('Only independent CityCode output maps may be rendered')
    STATE.update(job=job,started=time.monotonic(),finished=False,
                 report={'schema':'citycode.ue_visual_inspection.v1','status':'starting','map':level,
                         'engine_version':unreal.SystemLibrary.get_engine_version(),'ue_executed':True,
                         'captures':[],'maps_saved':False,'visual_review':'pending',
                         'automation_shader_methods':[name for name in dir(getattr(unreal,'AutomationLibrary',None))if 'shader'in name.lower()]})
    persist()
    try:
        current_world=unreal.EditorLevelLibrary.get_editor_world()
        current_level=current_world.get_path_name().split('.')[0] if current_world else None
        if job.get('require_current_map',False) and current_level!=level:
            raise ValueError('The current map differs from this inspection job; it was not changed')
        if current_level!=level and not unreal.EditorLevelLibrary.load_level(level):
            raise RuntimeError('Cannot load the independently saved CityCode map')
        world=unreal.EditorLevelLibrary.get_editor_world();STATE['world']=world
        box=bounds();STATE['report']['actual_world_bounds_m']=box
        STATE['views']=job.get('views')or default_views(box)
        if job.get('inspection_daylight',job.get('mode')=='viewport'):
            STATE['report']['inspection_environment']=inspection_daylight()
            persist()
        if job.get('mode')=='viewport':
            view=STATE['views'][0];location=view['location_m'];target=view['target_m']
            difference=[b-a for a,b in zip(location,target)]
            rotation=unreal.Rotator(pitch=math.degrees(math.atan2(difference[2],math.hypot(*difference[:2]))),
                                    yaw=math.degrees(math.atan2(difference[1],difference[0])),roll=0.)
            unreal.EditorLevelLibrary.set_level_viewport_camera_info(unreal.Vector(*(v*100 for v in location)),rotation)
            STATE['report'].update(status='desktop_viewport_positioned_visual_resources_pending',camera=view,maps_saved=False,
                                   captures=[],visual_review='interactive_user_review_required',
                                   visual_resources_ready=False,
                                   resource_status='UE may still be preparing shaders, textures, meshes and Lumen cards. Actor readback does not verify these resources.')
            unreal.log_warning('CityCode camera and inspection lighting are configured. Wait for UE shader/texture/mesh preparation before judging appearance; visual readiness is not yet verified.')
            persist();return
        for command in ('r.MotionBlurQuality 0','r.DepthOfFieldQuality 0','r.VSync 0','r.ScreenPercentage 100'):
            unreal.SystemLibrary.execute_console_command(world,command)
        if job.get('texture_pool_mib') is not None:
            unreal.SystemLibrary.execute_console_command(world,'r.Streaming.PoolSize '+str(int(job['texture_pool_mib'])))
        actor=unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SceneCapture2D,unreal.Vector(0,0,0))
        actor.set_actor_label('CityCode_TransientVisualInspection')
        component=actor.get_component_by_class(unreal.SceneCaptureComponent2D)
        component.set_editor_property('projection_type',unreal.CameraProjectionMode.PERSPECTIVE)
        component.set_editor_property('capture_on_movement',True)
        component.set_editor_property('always_persist_rendering_state',True)
        width=int(job.get('width',1280));height=int(job.get('height',960))
        target=unreal.RenderingLibrary.create_render_target2d(world,width,height,
                     unreal.TextureRenderTargetFormat.RTF_RGBA8_SRGB,unreal.LinearColor(.04,.05,.065,1.),False)
        component.set_editor_property('texture_target',target)
        STATE.update(actor=actor,component=component,target=target)
        STATE['report'].update(status='warming_up',resolution=[width,height],views=STATE['views'])
        setup_view(0)
        STATE['handle']=unreal.register_slate_post_tick_callback(tick)
    except Exception as error:
        unreal.log_error('CityCode visual QA setup failed: '+str(error));finish(error)


if __name__=='__main__':main()
