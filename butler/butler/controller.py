from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from butler.utils import *


'''
Convenient class for running AI2-THOR controller
'''


class ButController(Controller):
    def __init__(self, scene, **kwargs):
        super().__init__(scene=scene,
                         platform=CloudRendering,
                         renderInstanceSegmentation=True,
                         renderDepthImage=True,
                         renderNormalsImage=True,
                         renderSemanticSegmentation=True,
                         **kwargs)

        self.stored_frames = {}  # for rendering video,
        # self.stored_frames['camera_id'] = self.event.third_party_camera_frames[camera_id]
        # self.stored_frames['agent_camera'] = self.event.frame

    @property
    def event(self) -> ai2thor.server.Event:
        return self.last_event

    def show_video(self, camera='agent_camera', fps=10) -> None:
        """
        show the video of the stored frames
        """
        if camera in self.stored_frames:
            show_video(self.stored_frames[camera], fps=fps)

    def flush_stored_frames(self, camera='all') -> None:
        if camera == 'all':
            self.stored_frames = {}
        elif camera in self.stored_frames:
            del self.stored_frames[camera]

    def step(self, action, **kwargs) -> ai2thor.server.Event:
        """
        Step the environment but if stored_frame=True
        is passed as a keyword argument, store the frame
        """
        event = super().step(action, **kwargs)
        if 'stored_frame' in kwargs and kwargs['stored_frame']:
            camera = kwargs['camera'] if 'camera' in kwargs else 'agent_camera'
            if camera == 'agent_camera':
                frames = self.event.frame
            else:
                frames = self.event.third_party_camera_frames[int(camera)]
            self.stored_frames[camera].append(frames)
        return event

    @property
    def agent(self) -> Dict:
        '''
        Return the agent state
        e.g.,
        {'name': 'agent',
        'position': {'x': 0.0, 'y': 0.9009992480278015, 'z': -1.25},
        'rotation': {'x': -0.0, 'y': 90.0, 'z': 0.0},
        'cameraHorizon': -0.0,
        'isStanding': True,
        'inHighFrictionArea': False}
        '''
        return self.event.metadata['agent']

    def render(self, plot=True, axis_off=True) -> np.ndarray:
        '''
        Render the current scene
        if plot is True, plot the RGB frame
        return the RGB frame
        '''
        if plot:
            plt.imshow(self.event.frame)
            if axis_off:
                plt.axis('off')
        return self.event.frame

    def plot_frames(self) -> None:
        '''
        plot all the frames on an AI2-THOR Event
        including RGB, depth, instance segmentation, normals, and semantic segmentation
        '''
        plot_frames(self.event)

    @property
    def all_objects(self) -> pd.DataFrame:
        """
        get a panda frame of all objects in the floorplan 
        with their properties and metadata
        """
        return preprocess_objects(self.event.metadata['objects'])

    @property
    def objects(self) -> pd.DataFrame:
        """
        All objects in the current scene / view (in the visual range)
        """
        objects_in_view = list(self.event.instance_detections2D.keys())
        return self.all_objects[self.all_objects['objectId'].isin(objects_in_view)]

    @property
    def attributes(self) -> pd.Index:
        return self.objects.columns

    @property
    def probably_interactable_objects(self) -> pd.DataFrame:
        """
        get all the objects in self.objects that are:
        - visible: where visibility means NOT only the object is in the view
        but also it is within the visibility distance 
        (see: https://github.com/allenai/ai2thor/issues/815)
        and part of pixels more than certain percentage probably
        - isInteractable: it is not occluded by other objects
        """
        return self.objects[(self.objects['isInteractable']) & (self.objects['visible'])]

    @property
    def interactable_objects(self) -> pd.DataFrame:
        objs = self.probably_interactable_objects
        # filter objs to only include those where get_actionable_properties on the objectType column returns a non-empty list
        return objs[objs['objectType'].apply(lambda obj_type: len(get_actionable_properties(self.event, obj_type))) > 0]

    @property
    def uninteractable_objects(self) -> pd.DataFrame:
        """
        get all the objects in self.objects that are invisible
        """
        return self.objects[(self.objects['visible'] == False)]

    def objects_by_type(self, obj_type, only_visible=False,  only_probably_interactable=False,
                        only_interactable=False, ret_list=False) -> Union[pd.DataFrame, List]:
        """
        get all the objects in the scene by type
        if only_visible is True, only return the objects in the current view
        if only_probably_interactable is True, return the objects where the visible attribute is true (i.e., within the visibility distance)
        if only_interactbale is True, return the objects where the visible attribute is true and the object has actionable properties
        else return all the objects in the floorplan
        """
        objects = self.event.objects_by_type(obj_type)
        if only_visible:
            in_scene_objs = set(self.objects['name'])
            objects = [obj for obj in objects
                       if obj['name'] in in_scene_objs]
        if only_probably_interactable:
            probably_interactable_objs = set(
                self.probably_interactable_objects['name'])
            objects = [obj for obj in objects
                       if obj['name'] in probably_interactable_objs]
        if only_interactable:
            interactable_objs = set(self.interactable_objects['name'])
            objects = [obj for obj in objects
                       if obj['name'] in interactable_objs]

        if ret_list:
            return objects
        return preprocess_objects(objects)

    def get_obj_by_name(self, name) -> pd.DataFrame:
        """
        get the objectId of an object by its name
        """
        # search on dataframe objects
        return self.objects[self.objects['name'] == name]

    def get_obj_by_id(self, obj_id) -> pd.DataFrame:
        """
        get the object by its objectId
        """
        return self.objects[self.objects['objectId'] == obj_id]

    @property
    def object_types(self) -> set:
        """
        get all the object types in the scene
        """
        return get_object_types(self.objects)

    @property
    def all_object_types(self) -> set:
        return get_object_types(self.all_objects)

    def get_actionable_properties(self, object_type) -> List[str]:
        """
        get all the actionable properties of an object type
        in the scene
        """
        return get_actionable_properties(self.event, object_type)

    def isChangeTemp(self, obj_type) -> bool:
        """
        check if the object type is a heat or cold source
        """
        return isChangeTemp(self.event, obj_type)

    @classmethod
    def get_scenes(cls, scene_type) -> List[str]:
        """
        get all the scenes of a certain type
        """
        # create a fake controller to get all the scenes
        fake_controller = ButController(scene='FloorPlan1')
        return get_scenes(fake_controller, scene_type)

    def mask_object_type(self, object_type, plot=True, axis_off=True) -> np.ndarray:
        """
        Return a semantic mask of the object type in the current view
        """
        # get all objects of the object type
        objects = self.objects_by_type(object_type, ret_list=True)
        # get the instance_masks of all the objects of this type
        masks = self.event.instance_masks
        masked_img = np.sum([masks[obj['objectId']]
                            for obj in objects], axis=0)
        if plot:
            plt.imshow(masked_img, cmap='gray')
            if axis_off:
                plt.axis('off')

        return masked_img  # 0 and 1 mask where 1 means that pixel is part of the object

    def mask_object(self, objectId, plot=True, axis_off=True) -> np.ndarray:
        """
        Return an instance mask of the object in the current view
        """
        masked_img = self.event.instance_masks[objectId]
        if plot:
            plt.imshow(masked_img, cmap='gray')
            if axis_off:
                plt.axis('off')
        return masked_img

    def render_third_party_camera(self, camera_id=0, plot=True, axis_off=True) -> np.ndarray:
        if not self.event.third_party_camera_frames:
            return
        if plot:
            plt.imshow(self.event.third_party_camera_frames[camera_id])
        if axis_off:
            plt.axis('off')
        return self.event.third_party_camera_frames[camera_id]


class MultiButController(ButController):
    def __init__(self, scene, agentCount, **kwargs):
        super().__init__(scene, agentCount=agentCount, **kwargs)
        self.agentCount = agentCount
        self.current_agent_id = 0

    def choose_agent(self, agent_id) -> None:
        assert agent_id < self.agentCount, "agent_id must be less than agentCount"
        self.current_agent_id = agent_id

    @property
    def event(self) -> ai2thor.server.Event:
        return self.last_event.events[self.current_agent_id]

    def render_agents(self) -> List[np.ndarray]:
        """
        render all the agents
        """
        rgb_frames = [event.frame for event in self.last_event.events]
        visualize_frames(rgb_frames)
        return rgb_frames


class PrimitiveButController(ButController):
    def __init__(self, scene, **kwargs):
        super().__init__(scene=scene,
                         agentMode="arm",
                         **kwargs)
