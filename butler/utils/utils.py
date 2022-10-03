"""
Source: https://github.com/allenai/ai2thor-colab/blob/main/ai2thor_colab
"""
from typing import Sequence, Union, Optional,  List, Tuple, Dict
from PIL import Image
import ai2thor.server
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import ImageSequenceClip
import pandas as pd
from collections import OrderedDict


def preprocess_objects(objects):
    '''
    Reorder some columns (attributes) of each object to be logical (e.g., move the objectId to the front).
    '''
    processed_objects = []
    for obj in objects:
        obj = obj.copy()
        obj["position[x]"] = round(obj["position"]["x"], 4)
        obj["position[y]"] = round(obj["position"]["y"], 4)
        obj["position[z]"] = round(obj["position"]["z"], 4)

        obj["rotation[x]"] = round(obj["rotation"]["x"], 4)
        obj["rotation[y]"] = round(obj["rotation"]["y"], 4)
        obj["rotation[z]"] = round(obj["rotation"]["z"], 4)

        del obj["position"]
        del obj["rotation"]

        # these are too long to display
        del obj["objectOrientedBoundingBox"]
        del obj["axisAlignedBoundingBox"]
        del obj["receptacleObjectIds"]

        obj["mass"] = round(obj["mass"], 4)
        obj["distance"] = round(obj["distance"], 4)

        obj = OrderedDict(obj)
        obj.move_to_end("distance", last=False)
        obj.move_to_end("rotation[z]", last=False)
        obj.move_to_end("rotation[y]", last=False)
        obj.move_to_end("rotation[x]", last=False)

        obj.move_to_end("position[z]", last=False)
        obj.move_to_end("position[y]", last=False)
        obj.move_to_end("position[x]", last=False)

        obj.move_to_end("name", last=False)
        obj.move_to_end("objectId", last=False)
        obj.move_to_end("objectType", last=False)

        processed_objects.append(obj)
    # print(
    #     "Object Metadata. Not showing objectOrientedBoundingBox, axisAlignedBoundingBox, and receptacleObjectIds for clarity."
    # )
    return pd.DataFrame(processed_objects)


def plot_frames(event: Union[ai2thor.server.Event, np.ndarray]) -> None:
    """Visualize all the frames on an AI2-THOR Event.
    Example:
    plot_frames(controller.last_event)
    """
    if isinstance(event, ai2thor.server.Event):
        frames = dict()
        third_person_frames = event.third_party_camera_frames
        if event.frame is not None:
            frames["RGB"] = event.frame
        if event.instance_segmentation_frame is not None:
            frames["Instance Segmentation"] = event.instance_segmentation_frame
        if event.semantic_segmentation_frame is not None:
            frames["Semantic Segmentation"] = event.semantic_segmentation_frame
        if event.normals_frame is not None:
            frames["Normals"] = event.normals_frame
        if event.depth_frame is not None:
            frames["Depth"] = event.depth_frame

        if len(frames) == 0:
            raise Exception("No agent frames rendered on this event!")

        rows = 2 if len(third_person_frames) else 1
        cols = max(len(frames), len(third_person_frames))
        fig, axs = plt.subplots(
            nrows=rows, ncols=cols, dpi=150, figsize=(3 * cols, 3 * rows)
        )

        agent_row = axs[0] if rows > 1 else axs

        for i, (name, frame) in enumerate(frames.items()):
            ax = agent_row[i] if cols > 1 else agent_row
            im = ax.imshow(frame)
            ax.axis("off")
            ax.set_title(name)

            if name == "Depth":
                fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax)

        # set unused axes off
        for i in range(len(frames), cols):
            agent_row[i].axis("off")

        # add third party camera frames
        if rows > 1:
            for i, frame in enumerate(third_person_frames):
                ax = axs[1][i] if cols > 1 else axs[1]
                ax.imshow(frame)
                ax.axis("off")
            for i in range(len(third_person_frames), cols):
                axs[1][i].axis("off")

            fig.text(x=0.1, y=0.715, s="Agent Frames",
                     rotation="vertical", va="center")
            fig.text(
                x=0.1,
                y=0.3025,
                s="Third Person Frames",
                rotation="vertical",
                va="center",
            )
    elif isinstance(event, np.ndarray):
        return Image.fromarray(event)
    else:
        raise Exception(
            f"Unknown type: {type(event)}. "
            "Must be np.ndarray or ai2thor.server.Event."
        )


def show_video(frames: Sequence[np.ndarray], fps: int = 10):
    """Show a video composed of a sequence of frames.
    Example:
    frames = [
        controller.step("RotateRight", degrees=5).frame
        for _ in range(72)
    ]
    show_video(frames, fps=5)
    """
    frames = ImageSequenceClip(frames, fps=fps)
    return frames.ipython_display()


def get_object_types(objects):
    """
    Get a set of object types
    objects: list of objects
    """
    return set([obj["objectType"] for obj in objects])


# source: https://ai2thor.allenai.org/ithor/documentation/objects/object-types
ACTIONABLE_PROPERTIES = [
    "openable",  # drawer, cabinet (action: open / close)
    "pickupable",  # apple, box (action: pickup, drop, put, ect...)
    "moveable",  # chair, coffee table (action: push, pull)
    "toggleable",  # toggleOn / toggleOff (e.g., turning lamp on / off)
    # allow (some) other objects to be placed on them. (action: put)
    "receptacle",
    # "fillable",  # mug (action: fillWithCoffee or fillWithWater, emptyLiquid)
    "canFillWithLiquid",
    "sliceable",  # apple, bread (action: slice. Irreversible)
    # egg, bread (action: cook), these objects can be cooked implicitly
    "cookable",
    # by interacting with heat sources (e.g., microwave on), this action
    # is cheating a bit since we can only cook an egg on a pan for example
    # in real life
    "breakable",  # mug, glass (action: breakObject)
    "dirtyable",  # mug, glass (action: dirtyObject, cleanObject)
    "canBeUsedUp",  # toiletPaper (action: useUp)
]

MATERIAL_PROPERTIES = [
    "temperature",  # hot, cold, room temperature
    # "changeTemp",  # these objects can change temperature of other objects e.g., stoveBurner
    # # action like throw will be affected by mass (lighter objects will go further)
    "isHeatSource",  # stoveBurner, fire
    "isColdSource",  # fridge
    "mass",
    # bowls can be made of plastic, glass or ceramic, these objects might look slightly different
    "salientMaterials",
]

# actions
MOVES = [
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
]

ROTATES = [
    "RotateLeft",
    "RotateRight",
]

LOOK = [
    "lookUp",
    "lookDown",
]


def get_actionable_properties(event, obj_type):
    """
    Get a list of actionable properties for an object type in a scene.
    """
    # find one object of the given type
    try:
        obj = event.objects_by_type(obj_type)[0]
    except:
        print('No object of type {} found in the current scene'.format(obj_type))
        return []
    # check if actionable properties is true for the object
    return [prop for prop in ACTIONABLE_PROPERTIES if obj[prop]]


def isChangeTemp(event, obj_type):
    """
    Check if an object type is a heat or cold source.
    """
    # find one object of the given type
    obj = event.objects_by_type(event, obj_type)[0]
    return obj["isHeatSource"] or obj["isColdSource"]


def get_scenes(controller, scene_type):
    # COMMENT: controller.ithor_scenes should have been
    # implemented as a class method rather than instance
    # method
    assert scene_type in ['kitchen', 'bedroom', 'bathroom',
                          'living_room'], "scene_type must be one of kitchen, bedroom, bathroom, living_room"
    return controller.ithor_scenes(
        include_kitchens=scene_type == "kitchen",
        include_bedrooms=scene_type == "bedroom",
        include_bathrooms=scene_type == "bathroom",
        include_living_rooms=scene_type == "living_room",
    )


def side_by_side(
    frame1: np.ndarray, frame2: np.ndarray, title: Optional[str] = None,
    cmap_dict: Optional[Dict[int, str]] = None
) -> None:
    """Plot 2 image frames next to each other.
    Example:
    event1 = controller.last_event
    event2 = controller.step("RotateRight")
    overlay(event1.frame, event2.frame)
    """
    fig, axs = plt.subplots(nrows=1, ncols=2, dpi=150, figsize=(8, 5))
    # cmap[0] = optional cmap for axs[0], cmap[1] = optional cmap for axs[1]
    axs[0].imshow(
        frame1, cmap=None if not cmap_dict or 0 not in cmap_dict else cmap_dict[0])
    axs[0].axis("off")
    axs[1].imshow(
        frame2, cmap=None if not cmap_dict or 1 not in cmap_dict else cmap_dict[1])
    axs[1].axis("off")
    if title:
        fig.suptitle(title, y=0.85, x=0.5125)


def visualize_frames(
    rgb_frames: List[np.ndarray],
    title: str = '',
    figsize: Tuple[int, int] = (8, 2)
) -> plt.Figure:
    """Plots the rgb_frames for each agent."""
    fig, axs = plt.subplots(
        1, len(rgb_frames), figsize=figsize, facecolor='white', dpi=300)
    for i, frame in enumerate(rgb_frames):
        ax = axs[i]
        ax.imshow(frame)
        ax.set_title(f'AgentId: {i}')
        ax.axis('off')
    if title:
        fig.suptitle(title)
    return fig


def is_in(obj, objs):
    if isinstance(objs, list):
        pass
