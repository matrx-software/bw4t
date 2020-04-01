from itertools import product

from matrx import WorldBuilder
import numpy as np
from matrx.goals import LimitedTimeGoal
from matrx.grid_world import GridWorld
from matrx.objects import SquareBlock, AreaTile
from matrx.world_builder import RandomProperty
from matrx.goals import UseCaseGoal


def calculate_world_size(nr_rooms, room_size, hallway_space, nr_drop_zones, nr_blocks_needed, rooms_per_row):
    nr_room_rows = np.ceil(nr_rooms / rooms_per_row)

    # calculate the total width
    world_width = max(rooms_per_row * room_size[0] + 2 * hallway_space,
                      (nr_drop_zones + 1) * hallway_space + nr_drop_zones) + 2

    # calculate the total height
    world_height = nr_room_rows * room_size[1] + (nr_room_rows + 1) * hallway_space + nr_blocks_needed + 2

    return int(world_width), int(world_height)


def get_room_locations(room_loc, door_loc, room_size):
    # We always assume that locations where we can position blocks are not behind the door and as maximum as possible,
    # meaning that we will position blocks in rows with one gap in between.
    locs = []

    for y in range(door_loc[1] - 1, room_loc[1],
                   -2):  # all rows starting from the row closest to door and skipping one each time
        for x in range(room_loc[0] + 1, door_loc[0]):  # all locations left to door
            locs.append((x, y))

        for x in range(door_loc[0] + 1, room_loc[0] + room_size[0] - 1):  # all locations left to door
            locs.append((x, y))

    return locs


def create_builder():
    # Some general settings
    tick_duration = 0.2
    random_seed = 1
    verbose = False

    # Some BW4T settings
    room_size = (8, 8)  # width, height
    nr_rooms = 16
    rooms_per_row = 5
    average_blocks_per_room = 3
    block_shapes = [0, ]
    block_colors = ['#0008ff', '#ff1500', '#0dff00']
    room_colors = ['#0008ff', '#ff1500', '#0dff00']
    wall_color = "#8a8a8a"
    drop_off_color = "#878787"
    block_size = 0.5
    nr_drop_zones = 3
    nr_blocks_needed = 3
    hallway_space = 3

    # Set numpy's random generator
    np.random.seed(random_seed)

    # Get world size
    world_size = calculate_world_size(nr_rooms, room_size, hallway_space, nr_drop_zones, nr_blocks_needed,
                                      rooms_per_row)

    # Create the goal
    # TODO
    goal = LimitedTimeGoal(max_nr_ticks=10)

    # Get the world builder
    builder = WorldBuilder(shape=world_size, tick_duration=tick_duration, random_seed=random_seed, run_matrx_api=True,
                           run_matrx_visualizer=True, verbose=verbose, simulation_goal=goal)

    # Add the world bounds
    builder.add_room(top_left_location=(0, 0), width=world_size[0], height=world_size[1], name="world_bounds")

    # Create the rooms
    room_locations = {}
    for room_nr in range(nr_rooms):
        row = np.floor(room_nr / rooms_per_row)
        column = room_nr % rooms_per_row

        # x is: +1 for the edge, +edge hallway, +room width * column nr, +1 off by one
        room_x = int(1 + hallway_space + (room_size[0] * column) + 1)

        # y is: +1 for the edge, +hallway space * (nr row + 1 for the top hallway), +row * room height, +1 off by one
        room_y = int(1 + hallway_space * (row + 1) + row * room_size[1] + 1)

        # door location is always center bottom
        door_x = room_x + int(np.ceil(room_size[0] / 2))
        door_y = room_y + room_size[1] - 1

        # Select random room color
        np.random.shuffle(room_colors)
        room_color = room_colors[0]

        # Add the room
        room_name = f"room_{room_nr}"
        builder.add_room(top_left_location=(room_x, room_y), width=room_size[0], height=room_size[1], name=room_name,
                         door_locations=[(door_x, door_y)], wall_visualize_colour=wall_color,
                         with_area_tiles=True, area_visualize_colour=room_color, area_visualize_opacity=0.1)

        # Find all inner room locations where we allow objects (making sure that the location behind to door is free)
        block_locations = get_room_locations((room_x, room_y), (door_x, door_y), room_size)
        room_locations[room_name] = block_locations

    # Add the collectible objects, we do so probabilistically so each world would contain different blocks
    for room_name, locations in room_locations.items():
        for loc in locations:
            # Get the block's properties
            name = f"Block in {room_name}"

            # Get the probability so we have on average the requested number of blocks per room
            prob = min(1.0, average_blocks_per_room / len(locations))

            # Create a MATRX random property of shape and color so each world varies
            colour_property = RandomProperty(values=block_colors)
            shape_property = RandomProperty(values=block_shapes)

            # Add the block
            builder.add_object_prospect(loc, name, probability=prob, visualize_shape=shape_property,
                                        visualize_colour=colour_property, is_movable=True, visualize_size=block_size)

    # Create the drop-off zones, this includes generating the random colour/shape combinations to collect.
    x = int(np.ceil(world_size[0] / 2)) - (int(np.floor(nr_drop_zones / 2)) * (hallway_space + 1))
    y = world_size[1] - 1 - 1  # once for off by one, another for world bound
    for nr_zone in range(nr_drop_zones):
        # Add the zone's tiles
        builder.add_area((x, y - nr_blocks_needed + 1), width=1, height=nr_blocks_needed, name=f"Drop off {nr_zone}",
                         visualize_colour=drop_off_color, zone_nr=nr_zone)
        # Go through all needed blocks
        for nr_block in range(nr_blocks_needed):
            # Create a MATRX random property of shape and color so each world varies
            colour_property = RandomProperty(values=block_colors)
            shape_property = RandomProperty(values=block_shapes)

            # Add a 'ghost image' of the block that should be collected
            loc = (x, y - nr_block)
            builder.add_object(loc, name="Collect Block", visualize_colour=colour_property,
                               visualize_shape=shape_property, visualize_size=block_size)

        # Change the x to the next zone
        x = x + hallway_space + 1

    # Add the agents
    # TODO

    # Return the builder
    return builder


class CollectionGoal(UseCaseGoal):

    def __init__(self):
        super().__init__()

        # list of all drop of locations, each locations consisting of the tile's id and desirable block and current
        # block on that tile (if any)
        self.drop_off = None

    def goal_reached(self, grid_world: GridWorld):
        if self.drop_off is None:  # find all drop off locations and its tile ID's
            all_obj = grid_world.environment_objects
            for obj_id, obj in all_obj.items():
                pass
