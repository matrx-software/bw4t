from matrx.agents import AgentBrain


class BlockWorldAgent(AgentBrain):

    def __init__(self):
        super().__init__()

    def initialize(self):
        """ This method is called when the world is created. Use it for instance to send 'hello' messages to other
        agents."""
        pass

    def filter_observations(self, state):
        """ This method can further filter the `state`. This is a MATRX object that you can access using some helper
         methods or simply as a dictionary. It contains all perceived objects by this agent based on its
         `SenseCapability` it received when this agent was added to the `WorldBuilder`. These perceived objects are
         represented as a bunch of key-value pairs representing the object's properties (such as its color).

         An important property is the object ID (e.g. some_id=obj['obj_id']), as this is the unique identifier that
         MATRX uses to know which object is meant. For instance, you can use this id to access its own properties in
         the state (e.g., obj=state["some_id"]). You even need it to perform some actions (e.g., to pick up an object
         you need to provide the ID of that object so MATRX knows which object you refer to).

         This method can for example make an agent color blind by removing all color properties from the objects.

         Note that if this is the first time you are in this method, you can extract or initialize quite some useful
         stuff from the state. For example, you can initialize a dict of rooms telling you whether you visited them
         before and which objects you saw there when you did.

         I would really advise to see what `state` contains using a debugger or by printing all of its content:
            `print(state.__State_state_dict)`
         """
        return state

    def decide_on_action(self, state):
        """ The method where the actual reasoning happens. This should return the string name of an action
        (e.g. "GrabObject" or GrabObject.__name__ if the action is imported). It should also return any arguments to
         perform this action (e.g. the object ID for the object you wish to pickup).

         Note that MATRX offers a handy movement tool, called the `Navigator` which you can use to implement waypoint
         navigation.
         This link for its code: https://github.com/matrx-software/matrx/blob/09ac0c06f33001649a089a4a6d9a0d86ca2a2aa8/matrx/agents/agent_utils/navigator.py#L10
         THis link for an example: https://github.com/matrx-software/matrx/blob/09ac0c06f33001649a089a4a6d9a0d86ca2a2aa8/matrx/agents/agent_types/patrolling_agent.py#L6

         A naive BW4T agent could follow these steps (somewhat like a decision tree, so a lot of condition checking. It
         might be good to offer these as helper functions for the students):
         1) Check if the agent holds has an object that needs to be dropped in the drop zone (see the properties of
         this agent in state).
            1.1) If this is the case, find the location in state
            1.2) CHeck if you are at that location, if so drop the object
            1.3) If not at the location use the Navigator to get the next move action to get there
        2) If not holding the next object;
            2.1) Check what the next object is to collect by searching for the drop locations and 'ghost objects' in
            state.
            2.2) See if you already saw that object once in a room using a custom made variable in this agent
            2.3) If you already saw this object somewhere, move towards it using the Navigator (note; the navigator
            does not account for doors, so these need to be opened otherwise the path will be blocked)
            2.4) If you are at the location of the object, pick it up
        3) If you do not know about the next object;
            3.1) Go to the first door of a room you have not checked before (using another attribute that needs to be
            made in this agent, e.g., a dict of room names and booleans whether you visited them).
            3.2) Walk in that room and update your seen-blocks-list
        """
        return None, {}
