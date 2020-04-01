from collections import OrderedDict
from itertools import product

from matrx import WorldBuilder
import numpy as np
from matrx.actions import MoveNorth, OpenDoorAction, CloseDoorAction
from matrx.actions.move_actions import MoveEast, MoveSouth, MoveWest
from matrx.agents import AgentBrain, HumanAgentBrain
from matrx.goals import LimitedTimeGoal
from matrx.grid_world import GridWorld, DropObject, GrabObject
from matrx.objects import SquareBlock, AreaTile, EnvObject
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
    key_action_map = {  # For the human agents
        'w': MoveNorth.__name__,
        'd': MoveEast.__name__,
        's': MoveSouth.__name__,
        'a': MoveWest.__name__,
        'q': GrabObject.__name__,
        'e': DropObject.__name__,
        'r': OpenDoorAction.__name__,
        'f': CloseDoorAction.__name__,
    }

    # Some BW4T settings
    room_size = (5, 5)  # width, height
    nr_rooms = 3
    rooms_per_row = 3
    average_blocks_per_room = 3
    block_shapes = [0, ]
    block_colors = ['#0008ff', '#ff1500', '#0dff00']
    room_colors = ['#0008ff', '#ff1500', '#0dff00']
    wall_color = "#8a8a8a"
    drop_off_color = "#878787"
    block_size = 0.5
    nr_drop_zones = 1
    nr_blocks_needed = 3
    hallway_space = 2
    nr_teams = 1
    agents_per_team = 2
    human_agents_per_team = 1

    # Set numpy's random generator
    np.random.seed(random_seed)

    # Get world size
    world_size = calculate_world_size(nr_rooms, room_size, hallway_space, nr_drop_zones, nr_blocks_needed,
                                      rooms_per_row)

    # Create the goal
    goal = CollectionGoal()

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

    # Add the collectible objects, we do so probabilistically so each world will contain different blocks
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
                                        visualize_colour=colour_property, is_movable=True,
                                        visualize_size=block_size, is_block=True, is_traversable=True)

    # Create the drop-off zones, this includes generating the random colour/shape combinations to collect.
    x = int(np.ceil(world_size[0] / 2)) - (int(np.floor(nr_drop_zones / 2)) * (hallway_space + 1))
    y = world_size[1] - 1 - 1  # once for off by one, another for world bound
    for nr_zone in range(nr_drop_zones):
        # Add the zone's tiles
        builder.add_area((x, y - nr_blocks_needed + 1), width=1, height=nr_blocks_needed, name=f"Drop off {nr_zone}",
                         visualize_colour=drop_off_color, drop_zone_nr=nr_zone, is_drop_zone=True, is_goal_block=False)
        # Go through all needed blocks
        for nr_block in range(nr_blocks_needed):
            # Create a MATRX random property of shape and color so each world varies
            colour_property = RandomProperty(values=block_colors)
            shape_property = RandomProperty(values=block_shapes)

            # Add a 'ghost image' of the block that should be collected
            loc = (x, y - nr_block)
            builder.add_object(loc, name="Collect Block", visualize_colour=colour_property,
                               visualize_shape=shape_property, visualize_size=block_size,
                               visualize_opacity=0.4, visualize_depth=85, is_movable=False,
                               is_traversable=True, drop_zone_nr=nr_zone, is_drop_zone=False,
                               is_goal_block=True)

        # Change the x to the next zone
        x = x + hallway_space + 1

    # Add the agents and human agents to the top row of the world
    loc = (0, 1)  # we begin adding agents to the top left, x is zero because we add +1 each time we add an agent
    for team in range(nr_teams):
        team_name = f"Team {team}"
        # Add agents
        nr_agents = agents_per_team - human_agents_per_team
        for agent_nr in range(nr_agents):
            brain = BlockWorldAgent()
            loc = (loc[0] + 1, loc[1])
            builder.add_agent(loc, brain, team=team_name, name=f"Agent {agent_nr} in {team_name}")

        # Add human agents
        for human_agent_nr in range(human_agents_per_team):
            brain = HumanAgentBrain(max_carry_objects=1, grab_range=0, drop_range=0)
            loc = (loc[0] + 1, loc[1])
            builder.add_human_agent(loc, brain, team=team_name, name=f"Human {human_agent_nr} in {team_name}",
                                    key_action_map=key_action_map)

    # Return the builder
    return builder


class CollectionGoal(UseCaseGoal):

    def __init__(self):
        super().__init__()

        # A dictionary of all drop locations. The keys is the drop zone number, the value another dict.
        # This dictionary contains as key the rank of the to be collected object and as value the location
        # of where it should be dropped, the shape and colour of the block, and the tick number the correct
        # block was delivered. The rank and tick number is there so we can check if objects are dropped in
        # the right order.
        self.__drop_off = None

        # We also track the progress
        self.__progress = 0

    def goal_reached(self, grid_world: GridWorld):
        if self.__drop_off is None:  # find all drop off locations, its tile ID's and goal blocks
            self.__find_drop_off_locations(grid_world)

        # Go through each drop zone, and check if the blocks are there in the right order
        is_satisfied, progress = self.__check_completion(grid_world)

        # Progress in percentage
        self.__progress = progress / sum([len(goal_blocks) for goal_blocks in self.__drop_off.values()])

        return is_satisfied

    def __find_drop_off_locations(self, grid_world):

        goal_blocks = {}  # dict with as key the zone nr and values list of ghostly goal blocks
        all_objs = grid_world.environment_objects
        for obj_id, obj in all_objs.items():  # go through all objects
            if "drop_zone_nr" in obj.properties.keys():  # check if the object is part of a drop zone
                zone_nr = obj.properties["drop_zone_nr"]  # obtain the zone number
                if obj.properties["is_goal_block"]:  # check if the object is a ghostly goal block
                    if zone_nr in goal_blocks.keys():  # create or add to the list
                        goal_blocks[zone_nr].append(obj)
                    else:
                        goal_blocks[zone_nr] = [obj]

        self.__drop_off = {}
        for zone_nr in goal_blocks.keys():  # go through all drop of zones and fill the drop_off dict
            # Instantiate the zone's dict.
            self.__drop_off[zone_nr] = {}

            # Obtain the zone's goal blocks.
            blocks = goal_blocks[zone_nr].copy()

            # The number of blocks is the maximum the max number blocks to collect for this zone.
            max_rank = len(blocks)

            # Find the 'bottom' location
            bottom_loc = (-np.inf, -np.inf)
            for block in blocks:
                if block.location[1] > bottom_loc[1]:
                    bottom_loc = block.location

            # Now loop through blocks lists and add them to their appropriate ranks
            for rank in range(max_rank):
                loc = (bottom_loc[0], bottom_loc[1] - rank)

                # find the block at that location
                for block in blocks:
                    if block.location == loc:
                        # Add to self.drop_off
                        self.__drop_off[zone_nr][rank] = [loc, block.visualize_shape, block.visualize_colour, None]

    def __check_completion(self, grid_world):
        # Get the current tick number
        curr_tick = grid_world.current_nr_ticks

        # loop through all zones, check the blocks and set the tick if satisfied
        for zone_nr, goal_blocks in self.__drop_off.items():
            # Go through all ranks of this drop off zone
            for rank, block_data in goal_blocks.items():
                loc = block_data[0]  # the location, needed to find blocks here
                shape = block_data[1]  # the desired shape
                colour = block_data[2]  # the desired colour
                tick = block_data[3]

                # Retrieve all objects, the object ids at the location and obtain all BW4T Blocks from it
                all_objs = grid_world.environment_objects
                obj_ids = grid_world.get_objects_in_range(loc, object_type=EnvObject, sense_range=0)
                blocks = [all_objs[obj_id] for obj_id in obj_ids
                          if obj_id in all_objs.keys() and "is_block" in all_objs[obj_id].properties.keys()]

                # Check if there is a block, and if so if it is the right one and the tick is not yet set, then set the
                # current tick.
                if len(blocks) > 0 and blocks[0].visualize_shape == shape and blocks[0].visualize_colour == colour and \
                        tick is None:
                    self.__drop_off[zone_nr][rank][3] = curr_tick
                # if there is no block, reset its tick to None
                elif len(blocks) == 0:
                    self.__drop_off[zone_nr][rank][3] = None

        # Now check if all blocks are collected in the right order
        is_satisfied = True
        progress = 0
        for zone_nr, goal_blocks in self.__drop_off.items():
            zone_satisfied = True
            ticks = [goal_blocks[r][3] for r in range(len(goal_blocks))]  # list of ticks in rank order

            # check if all ticks are increasing
            for idx, tick in enumerate(ticks[:-1]):
                if tick is None or ticks[idx+1] is None or not tick < ticks[idx+1]:
                    progress += (idx+1) if tick is not None else idx  # increment progress
                    zone_satisfied = False  # zone is not complete or ordered
                    break  # break this loop

            # if all ticks were increasing, check if the last tick is set and set progress to full for this zone
            if zone_satisfied and ticks[-1] is not None:
                progress += len(goal_blocks)

            # update our satisfied boolean
            is_satisfied = is_satisfied and zone_satisfied

        return is_satisfied, progress


class BlockWorldAgent(AgentBrain):

    def __init__(self):
        super().__init__()

    def initialize(self):
        pass

    def filter_observations(self, state):
        return state

    def decide_on_action(self, state):
        return None, {}
