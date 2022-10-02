from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
# import from utils
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

    @property
    def event(self):
        return self.last_event

    @property
    def agent(self):
        return self.event.metadata['agent']

    def render(self, plot=True, axis_off=True):
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

    def plot_frames(self):
        '''
        plot all the frames on an AI2-THOR Event
        including RGB, depth, instance segmentation, normals, and semantic segmentation
        '''
        plot_frames(self.event)

    @property
    def visible_objects(self):
        """
        get all the objects in self.objects that are visible
        where visibility means not only the object is in the view
        but also it is within the visibility distance so we can interact with it
        (see: https://github.com/allenai/ai2thor/issues/815)
        """
        return self.objects[self.objects['visible']]

    @property
    def invisible_objects(self):
        """
        get all the objects in self.objects that are invisible
        """
        return self.objects[(self.objects['visible'] == False)]

    @property
    def objects_in_floor(self):
        """
        get a panda frame of all objects in the floorplan 
        with their properties and metadata
        """
        return get_objects(self.event)

    @property
    def objects(self):
        """
        All objects in the current scene / view.
        """
        objects_in_view = list(self.event.instace_detections2D.keys())
        return self.objects_in_floor[self.objects_in_floor['objectId'].isin(objects_in_view)]

    @property
    def attributes(self):
        return self.objects.columns

    def get_obj_by_name(self, name):
        """
        get the objectId of an object by its name
        """
        # search on dataframe objects
        return self.objects[self.objects['name'] == name]

    def get_obj_by_id(self, obj_id):
        """
        get the object by its objectId
        """
        return self.objects[self.objects['objectId'] == obj_id]

    @property
    def object_types(self):
        """
        get all the object types in the scene
        """
        return get_object_types(self.event)

    def get_actionable_properties(self, object_type):
        """
        get all the actionable properties of an object type
        in the scene
        """
        return get_actionable_properties(self.event, object_type)

    def isChangeTemp(self, obj_type):
        """
        check if the object type is a heat or cold source
        """
        return isChangeTemp(self.event, obj_type)

    @classmethod
    def get_scenes(cls, scene_type):
        """
        get all the scenes of a certain type
        """
        # create a fake controller to get all the scenes
        fake_controller = ButController(scene='FloorPlan1')
        return get_scenes(fake_controller, scene_type)

    def focus_object_type(self, object_type, plot=True, axis_off=True):
        """
        focus on a certain object type in the scene
        """
        # get all objects of the object type
        objects = self.event.objects_by_type(object_type)
        # get the instance_masks of all the objects of this type
        masks = self.event.instance_masks
        masked_img = np.sum([masks[obj['objectId']]
                            for obj in objects], axis=0)
        if plot:
            plt.imshow(masked_img, cmap='gray')
            if axis_off:
                plt.axis('off')

        return masked_img  # 0 and 1 mask where 1 means that pixel is part of the object


class MultiButController(ButController):
    def __init__(self, scene, agentCount, **kwargs):
        super().__init__(scene, agentCount=agentCount, **kwargs)
        self.agentCount = agentCount
        self.current_agent_id = 0

    def choose_agent(self, agent_id):
        assert agent_id < self.agentCount, "agent_id must be less than agentCount"
        self.current_agent_id = agent_id

    @property
    def event(self):
        return self.last_event.events[self.current_agent_id]

    def render_agents(self):
        """
        render all the agents
        """
        rgb_frames = [event.frame for event in self.last_event.events]
        visualize_frames(rgb_frames)
        return rgb_frames
