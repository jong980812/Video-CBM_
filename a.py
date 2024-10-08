import json

# JSON data as provided by the user
json_data = {
    "0": [
        "Camera pointing at an object",
        "Walking towards the subject",
        "Camera zooming in on the target",
        "Adjusting camera settings while approaching"
    ],
    "1": [
        "Attaching a document to an email",
        "Fixing a picture to a wall",
        "Connecting a hose to a faucet",
        "Fastening a button onto a shirt"
    ],
    "2": [
        "Bending metal object",
        "Deformation",
        "Twisting",
        "Applying force"
    ],
    "3": [
        "Bending object",
        "Applying force",
        "Breaking point",
        "Destruction"
    ],
    "4": [
        "Digging a hole in the ground",
        "Placing an object into a hole",
        "Covering the object with soil",
        "Creating a hidden or buried item"
    ],
    "5": [
        "door",
        "window",
        "laptop lid",
        "book cover"
    ],
    "6": [
        "wrapping a gift with wrapping paper",
        "placing a blanket over a furniture",
        "concealing a book with a book cover",
        "shielding a surface with a tablecloth"
    ],
    "7": [
        "Shovel",
        "Dirt",
        "Hole",
        "Excavating"
    ],
    "8": [
        "Object falling behind another object",
        "Item slipping out of sight",
        "Item disappearing behind an obstruction",
        "Object dropping out of view"
    ],
    "9": [
        "Hand dropping a book in front of a table",
        "Object falling from someone's hand onto a floor",
        "An item being released and landing in front of another object",
        "A person accidentally letting go of an object in front of a counter"
    ],
    "10": [
        "hand releasing object",
        "object falling into container",
        "gravity taking effect",
        "transfer of object from hand to container"
    ],
    "11": [
        "dropped object",
        "adjacent object",
        "falling motion",
        "proximity"
    ],
    "12": [
        "hand holding an object",
        "object falling through the air",
        "object hitting another object",
        "object landing on a surface"
    ],
    "13": [
        "object too large",
        "inadequate space",
        "mismatched sizes",
        "improper fitting"
    ],
    "14": [
        "folded clothes",
        "wrinkled fabric",
        "folded paper",
        "crisp edges"
    ],
    "15": [
        "Baseball bat hitting a ball",
        "Hammer hitting a nail",
        "Boxer punching a bag",
        "Tennis racket hitting a ball"
    ],
    "16": [
        "Holding a book",
        "Holding a cup",
        "Holding a phone",
        "Holding a pen"
    ],
    "17": [
        "Hand holding package behind wall",
        "Concealed object behind curtain",
        "Person hiding object behind bookshelf",
        "Hand hiding something behind back"
    ],
    "18": [
        "hand holding a book in front of a shelf",
        "cup being held in front of a table",
        "phone held in front of a window",
        "hand holding a pen in front of a paper"
    ],
    "19": [
        "Hand holding a cup next to a saucer",
        "Hand holding a book next to a table",
        "Hand holding a phone next to an ear",
        "Hand holding a brush next to a canvas"
    ],
    "20": [
        "Hand holding a cup over a table",
        "Arm extending over a plate",
        "Fingers gripping a fork over a plate",
        "Hand placing a lid on a pot"
    ],
    "21": [
        "cup on its side",
        "book on its side",
        "plate on its side",
        "box on its side"
    ],
    "22": [
        "Rolling object",
        "Flat surface",
        "Motion",
        "Friction"
    ],
    "23": [
        "Rolling ball",
        "Slanted surface",
        "Gravity",
        "Potential energy"
    ],
    "24": [
        "Slanted surface",
        "Rolling object",
        "Gravity",
        "Motion"
    ],
    "25": [
        "Lifting a plate with food on it",
        "Lifting a book with a bookmark inside",
        "Lifting a tray with empty glasses on it",
        "Lifting a laptop with a charger plugged in"
    ],
    "26": [
        "Lifting surface",
        "Object on surface",
        "Sliding down",
        "Action"
    ],
    "27": [
        "Lifting a heavy box with both hands",
        "Holding an object above the head",
        "Elevating an item using a crane",
        "Hoisting a weight with a pulley system"
    ],
    "28": [
        "Lifting an object with arms extended",
        "Full range of motion in lifting an object",
        "Releasing an object from a raised position",
        "Gravity causing the object to drop down"
    ],
    "29": [
        "lifting heavy box",
        "using forklift",
        "moving large object",
        "using crane"
    ],
    "30": [
        "Lifting up one end of a book",
        "Holding up a corner of a table",
        "Raising a portion of a rug",
        "Lifting the edge of a carpet"
    ],
    "31": [
        "Object being lifted",
        "Gravity",
        "Motion",
        "Impact"
    ],
    "32": [
        "Zooming out",
        "Receding from the subject",
        "Pulling back from the scene",
        "Increasing the distance between the camera and the object"
    ],
    "33": [
        "Rotating gear",
        "Swinging pendulum",
        "Spinning wheel",
        "Moving conveyor belt"
    ],
    "34": [
        "Pushing object",
        "Sliding object",
        "Tilting object",
        "Toppling object"
    ],
    "35": [
        "Pushing an object",
        "Sliding an object",
        "Dragging an object",
        "Moving an object smoothly"
    ],
    "36": [
        "Pulling apart",
        "Separating objects",
        "Stretching out",
        "Creating distance"
    ],
    "37": [
        "Pushing an object towards another",
        "Dragging an item closer to another",
        "Sliding something nearer to something else",
        "Moving an object to bring it closer to another"
    ],
    "38": [
        "Pushing a box into a wall",
        "Throwing a ball at a target",
        "Slamming two doors together",
        "Swinging a hammer to hit a nail"
    ],
    "39": [
        "Pushing a cart past a shelf",
        "Sliding a stack of papers across a desk",
        "Moving a ladder through a doorway",
        "Pulling a suitcase along a conveyor belt"
    ],
    "40": [
        "Pushing object",
        "Pulling object",
        "Dragging object",
        "Lifting object"
    ],
    "41": [
        "Pushing an object away",
        "Pulling an object towards",
        "Moving object in opposite direction",
        "Adjusting object position"
    ],
    "42": [
        "Moving object towards another object",
        "Pulling object closer to another object",
        "Pushing object towards another object",
        "Bringing object nearer to another object"
    ],
    "43": [
        "Lowering an object",
        "Pushing something downwards",
        "Moving an item in a downward direction",
        "Bringing an object down"
    ],
    "44": [
        "Object being pushed",
        "Object approaching",
        "Motion towards the viewer",
        "Object moving closer"
    ],
    "45": [
        "Lifting a heavy object",
        "Raising an item",
        "Elevating a load",
        "Hoisting something upwards"
    ],
    "46": [
        "Opening a door",
        "Unboxing a package",
        "Opening a bottle",
        "Opening a book"
    ],
    "47": [
        "Hand reaching for an object",
        "Grasping an item",
        "Lifting an object off the ground",
        "Object being taken off a surface"
    ],
    "48": [
        "Stacking items",
        "Creating a pile",
        "Building a tower",
        "Arranging objects in a stack"
    ],
    "49": [
        "Electric plug",
        "Wall socket",
        "Power cord",
        "Connecting device"
    ],
    "50": [
        "Plugging",
        "Pulling out",
        "Hand",
        "Removing"
    ],
    "51": [
        "Poking a hole into dough",
        "Poking a hole into a piece of paper",
        "Poking a hole into a watermelon",
        "Poking a hole into clay"
    ],
    "52": [
        "hand holding a pointed object",
        "pushing the pointed object into a surface",
        "creating a hole in the surface",
        "surface being pierced by the pointed object"
    ],
    "53": [
        "Poking",
        "Stack",
        "Collapses",
        "Jenga"
    ],
    "54": [
        "Poking a stack of cards without the stack collapsing",
        "Poking a stack of books without the stack collapsing",
        "Poking a stack of pancakes without the stack collapsing",
        "Poking a stack of plates without the stack collapsing"
    ],
    "55": [
        "finger poking a balloon",
        "pushing a door slightly",
        "poking a pillow",
        "gently tapping a glass"
    ],
    "56": [
        "Poking",
        "Gentle",
        "Delicate",
        "Subtle"
    ],
    "57": [
        "Poking a vase on a table",
        "Knocking a tower of blocks",
        "Pushing a cup off the edge",
        "Tipping over a bookshelf"
    ],
    "58": [
        "Finger poking a spinning top",
        "Rotating object with a single touch",
        "Turning things with a sharp prod",
        "Causing objects to rotate by prodding them"
    ],
    "59": [
        "Liquid being poured into a glass",
        "Milk poured into a cereal bowl",
        "Water flowing into a pitcher",
        "Coffee being poured into a mug"
    ],
    "60": [
        "Pouring liquid into a glass until it spills",
        "Overflowing liquid from a container",
        "Liquid overflowing from a cup",
        "Pouring liquid and it goes beyond the brim"
    ],
    "61": [
        "Pouring liquid into a glass",
        "Pouring milk into a bowl of cereal",
        "Pouring sauce onto a plate of pasta",
        "Pouring hot water into a teapot"
    ],
    "62": [
        "Pouring liquid out of a glass",
        "Emptying a jar or bottle",
        "Spilling liquid from a container",
        "Draining liquid from a pitcher"
    ],
    "63": [
        "Hand reaching towards surface",
        "Failing to make contact",
        "Repeated wiping motion",
        "Expression of frustration"
    ],
    "64": [
        "Struggling with a stubborn jar lid",
        "Attempting to untie a tight knot",
        "Straining to open a locked door",
        "Trying to unscrew a tightly fastened bolt"
    ],
    "65": [
        "Pretending",
        "Tearing",
        "Non-tearable object",
        "Acting"
    ],
    "66": [
        "Hand hovering over a door handle",
        "Fingers near the edge of a drawer",
        "Unlatched suitcase with the lock unengaged",
        "Unfastened button on a shirt"
    ],
    "67": [
        "hand motion",
        "grasping an imaginary object",
        "twisting motion",
        "look of anticipation"
    ],
    "68": [
        "Hand reaching towards the ground",
        "Fingers mimicking the action of picking up",
        "Empty palm facing downwards",
        "Bending down towards an imaginary object"
    ],
    "69": [
        "Finger pointed towards an object",
        "Expression of curiosity or playfulness",
        "Hand moving in a poking motion",
        "Focused gaze towards the object being poked"
    ],
    "70": [
        "Pretending to pour from an empty teapot",
        "Acting like pouring water from an empty jug",
        "Simulating pouring liquid from an empty bottle",
        "Imaginary pouring from an empty pitcher"
    ],
    "71": [
        "Hand reaching behind an object",
        "Object being concealed",
        "Fingers clasping an item",
        "Motion of hiding an object"
    ],
    "72": [
        "Hand extending towards an object",
        "Object being held in mid-air",
        "Hand approaching a container",
        "Hand mimicking the action of putting something inside a container"
    ],
    "73": [
        "Faking the act of placing an object beside another object",
        "Simulating the action of putting something next to something",
        "Imitating the motion of positioning an item alongside another item",
        "Pretending to place an object adjacent to another object"
    ],
    "74": [
        "Hand hovering over a table surface",
        "Fingers mimicking placing an object",
        "Empty hand reaching towards a table",
        "Gesturing as if gently putting something on a surface"
    ],
    "75": [
        "Hand hovering above object",
        "Fingers mimicking placing",
        "Gentle touch without contact",
        "Gesturing towards object"
    ],
    "76": [
        "Hand placing object under table",
        "Eyes focused on object placement",
        "Slight smile on face",
        "Cautious and deliberate movement of hands"
    ],
    "77": [
        "scooping up sand with a shovel",
        "pretending to grab a ball with a fishing net",
        "miming scooping up water with a cupped hand",
        "acting as if capturing butterflies with a butterfly net"
    ],
    "78": [
        "Hand motion",
        "Blowing gesture",
        "Imaginary object",
        "Ephemeral action"
    ],
    "79": [
        "Hand motion",
        "Imaginary dust",
        "Sprinkle gesture",
        "Air magic"
    ],
    "80": [
        "Hand squeezing an invisible object",
        "Fingers applying pressure as if squeezing something",
        "Gripping motion with empty hands",
        "Playfully mimicking the act of squeezing an object"
    ],
    "81": [
        "reaching into an imaginary drawer",
        "grabbing an invisible object",
        "mimicking taking something from a shelf",
        "pretending to pick up an item from a table"
    ],
    "82": [
        "hand reaching into pocket",
        "object being pulled out from bag",
        "gesture of surprise",
        "action of pretending to hold an invisible object"
    ],
    "83": [
        "Throwing motion",
        "Hand gesture",
        "Faking a throw",
        "Deceptive action"
    ],
    "84": [
        "Hand holding an object upside down",
        "Object being rotated in the air",
        "Upside down position of the object",
        "Gesturing to flip an object"
    ],
    "85": [
        "Pulling a chair from behind a table",
        "Removing a curtain from behind a window",
        "Taking a book from behind a shelf",
        "Dragging a suitcase from behind a car"
    ],
    "86": [
        "Pulling a rope",
        "Moving a curtain",
        "Dragging a suitcase",
        "Sliding a drawer"
    ],
    "87": [
        "Pulling rope",
        "Moving object on a string",
        "Dragging something across a surface",
        "Tugging an item in a leftward direction"
    ],
    "88": [
        "Hand pulling curtain",
        "Dragging furniture across floor",
        "Pulling rope through pulley system",
        "Moving heavy object with a chain"
    ],
    "89": [
        "pulling object out of container",
        "retrieving something from a bag",
        "taking out an item from a pocket",
        "removing an object from a drawer"
    ],
    "90": [
        "Tug of war",
        "Stuck",
        "Resistance",
        "No movement"
    ],
    "91": [
        "tension",
        "stretching",
        "elasticity",
        "strained"
    ],
    "92": [
        "Pulling apart",
        "Separating",
        "Breaking in two",
        "Tearing"
    ],
    "93": [
        "Moving object horizontally",
        "Applying force to move an object",
        "Transferring something from left to right",
        "Exerting pressure to shift an item"
    ],
    "94": [
        "Pushing object",
        "Right to left movement",
        "Applied force",
        "Object displacement"
    ],
    "95": [
        "Pushing object",
        "Off the edge",
        "Forceful motion",
        "Displacement"
    ],
    "96": [
        "Box",
        "Surface",
        "Force",
        "Direction"
    ],
    "97": [
        "Spinning top",
        "Rotating object",
        "Circular motion",
        "Forceful rotation"
    ],
    "98": [
        "pushing an object forcefully",
        "leaning over an object to prevent it from falling",
        "applying pressure to stabilize an object",
        "nudging an object to the edge of a surface"
    ],
    "99": [
        "Falling object",
        "Table edge",
        "Pushing hand",
        "Displacement"
    ],
    "100": [
        "hand pushing an object",
        "object slightly moving",
        "force applied to object",
        "object being displaced"
    ],
    "101": [
        "Pushing box with hand",
        "Moving object with foot",
        "Using stick to push cart",
        "Pushing button with finger"
    ],
    "102": [
        "Counting objects",
        "Arranging items in a sequence",
        "Placing numbers onto a surface",
        "Organizing numerals onto an object"
    ],
    "103": [
        "Putting book on the table",
        "Placing cup on the table",
        "Setting plate on the table",
        "Arranging cutlery on the table"
    ],
    "104": [
        "Object being placed behind another object",
        "Rearranging objects",
        "Concealing an item",
        "Creating a hidden location"
    ],
    "105": [
        "Placing an object in front of another object",
        "Positioning an item in front of something else",
        "Arranging an item in front of another object",
        "Setting an object in front of a different object"
    ],
    "106": [
        "Inserting object",
        "Placing something inside",
        "Putting an item into a container",
        "Adding an object to a receptacle"
    ],
    "107": [
        "Placement of an object beside another object",
        "Arranging something alongside something",
        "Positioning an item next to another item",
        "Placing an object adjacent to another object"
    ],
    "108": [
        "Placing object horizontally",
        "Ensuring stability on the surface",
        "Preventing object from rolling",
        "Maintaining object's position on a flat surface"
    ],
    "109": [
        "Placing an object on a table",
        "Stacking items on a shelf",
        "Arranging objects on a countertop",
        "Putting a book on a bookshelf"
    ],
    "110": [
        "balancing object on edge",
        "unstable placement",
        "precarious balance",
        "inevitable fall"
    ],
    "111": [
        "Slanted surface",
        "Object placement",
        "Stable position",
        "Anti-slip surface"
    ],
    "112": [
        "Placing object on a surface",
        "Transferring item onto another object",
        "Arranging item onto a designated area",
        "Putting a thing onto a specific location"
    ],
    "113": [
        "placing",
        "unstable",
        "toppling",
        "unbalanced"
    ],
    "114": [
        "Arranging objects on the table",
        "Grouping similar items together",
        "Creating a cohesive display on the table",
        "Organizing objects based on their similarities"
    ],
    "115": [
        "Object rolling down a slanted surface",
        "Hand placing object on a slanted surface",
        "Object sliding down a slanted surface",
        "Slanted surface with object sliding down"
    ],
    "116": [
        "Putting object on slanted surface",
        "Object staying in place",
        "Creating stability on slanted surface",
        "Preventing object from rolling off"
    ],
    "117": [
        "Table",
        "Upright object",
        "Falling",
        "Sideways"
    ],
    "118": [
        "putting object on a table",
        "placing item in a drawer",
        "storing things in a cabinet",
        "laying something under a blanket"
    ],
    "119": [
        "cup",
        "mug",
        "bottle",
        "vase"
    ],
    "120": [
        "Putting",
        "something",
        "something",
        "on the table"
    ],
    "121": [
        "Peeling off a sticker",
        "Unveiling a surprise",
        "Taking off a cover",
        "Lifting a curtain"
    ],
    "122": [
        "rolling ball on table",
        "spinning toy car on floor",
        "pushing marble on countertop",
        "moving dice on game board"
    ],
    "123": [
        "Scooping ice cream with a spoon",
        "Scooping sand with a shovel",
        "Scooping rice into a bowl with a ladle",
        "Scooping water with a cup"
    ],
    "124": [
        "Holding a photo",
        "Presenting an image",
        "Displaying a picture",
        "Sharing a visual"
    ],
    "125": [
        "Revealing a hidden object",
        "Unveiling a secret",
        "Exposing something concealed",
        "Displaying an object behind an obstruction"
    ],
    "126": [
        "Hand holding a book next to a pair of glasses",
        "Finger pointing towards a sign next to a building",
        "Hand displaying a document next to a laptop",
        "Arm reaching out to present an object next to a table"
    ],
    "127": [
        "Placing an object on a surface",
        "Demonstrating an item on top of another object",
        "Presenting something above another thing",
        "Pointing at an object positioned above something else"
    ],
    "128": [
        "holding an object towards the camera",
        "pointing towards an object in front of the camera",
        "gesturing towards the camera",
        "presenting an item to the camera"
    ],
    "129": [
        "Empty container",
        "Vacant space",
        "Hollow object",
        "Clear area"
    ],
    "130": [
        "Container filled with objects",
        "Opening a box to reveal contents",
        "Putting something into a bag",
        "Unveiling a surprise hidden inside"
    ],
    "131": [
        "ball bouncing off a wall",
        "bullet ricocheting off a surface",
        "light beam reflecting off a mirror",
        "frisbee rebounding from a hand"
    ],
    "132": [
        "Two cars crashing and bouncing off each other",
        "A ball hitting a wall and bouncing back",
        "Two billiard balls colliding and changing direction",
        "A tennis racket hitting the ball and sending it in a different direction"
    ],
    "133": [
        "Car crashing into a wall",
        "Two billiard balls colliding",
        "Soccer player tackling another player",
        "Train hitting a barrier"
    ],
    "134": [
        "Feather falling gracefully",
        "Paper floating through the air",
        "Gentle descent of a feather",
        "Fluttering paper slowly descending"
    ],
    "135": [
        "falling object",
        "rock plummeting",
        "free-falling rock",
        "descending projectile"
    ],
    "136": [
        "Spilled liquid",
        "Container",
        "Messy surface",
        "Hidden spillage"
    ],
    "137": [
        "Spilled liquid near a cup",
        "Overflowing liquid beside a plate",
        "Liquids spilling close to a bottle",
        "Accidental spill next to a container"
    ],
    "138": [
        "Spilled liquid on a table",
        "Liquid dripping from a cup",
        "Stain spreading on a fabric",
        "Overflowing liquid from a container"
    ],
    "139": [
        "Spinning top",
        "Rotating object",
        "Continuous motion",
        "Circular movement"
    ],
    "140": [
        "Fidget spinner spinning",
        "Rapid rotation",
        "Abrupt halt in spinning",
        "Deceleration of spinning object"
    ],
    "141": [
        "Spreading butter onto bread",
        "Applying glue onto paper",
        "Pouring sauce onto pasta",
        "Brushing oil onto a skillet"
    ],
    "142": [
        "sprinkling salt onto food",
        "sprinkling glitter onto a craft project",
        "sprinkling water onto plants",
        "sprinkling sugar onto a dessert"
    ],
    "143": [
        "Squeezing a lemon",
        "Squeezing a stress ball",
        "Squeezing a tube of toothpaste",
        "Squeezing a hand sanitizer bottle"
    ],
    "144": [
        "Stack of books",
        "Pile of plates",
        "Tower of blocks",
        "Heap of toys"
    ],
    "145": [
        "Inserting item into container",
        "Pushing object into another",
        "Placing something inside",
        "Filling one thing with another"
    ],
    "146": [
        "Choosing an item from a selection on the table",
        "Picking one of many similar objects from the table",
        "Selecting an item from a variety of options on the table",
        "Deciding on one of the many similar objects placed on the table"
    ],
    "147": [
        "reaching for an object",
        "grasping an item",
        "lifting an object",
        "removing something"
    ],
    "148": [
        "Taking an item out of a box",
        "Removing an object from a bag",
        "Pulling something out of a pocket",
        "Extracting an item from a container"
    ],
    "149": [
        "Torn paper",
        "Ripped fabric",
        "Split object",
        "Torn document"
    ],
    "150": [
        "Torn paper edge",
        "Partially damaged object",
        "Visible tear lines",
        "Slight ripping of material"
    ],
    "151": [
        "Baseball being thrown",
        "Frisbee being launched",
        "Basketball being tossed",
        "Paper airplane being thrown"
    ],
    "152": [
        "Baseball player throwing a ball towards the outfield",
        "Basketball player throwing a ball towards the hoop",
        "Child throwing a frisbee at a park",
        "Athlete throwing a javelin in a competition"
    ],
    "153": [
        "Throwing",
        "Object",
        "Air",
        "Catching"
    ],
    "154": [
        "Throwing",
        "Object",
        "Air",
        "Fall"
    ],
    "155": [
        "Throwing ball",
        "Surface",
        "Projectile motion",
        "Forceful action"
    ],
    "156": [
        "tilting",
        "balancing",
        "safeguarding",
        "preventing from falling"
    ],
    "157": [
        "tilting object",
        "balancing item",
        "gradual movement",
        "displacement"
    ],
    "158": [
        "Cup falling over",
        "Bottle being knocked over",
        "Plate sliding off the table",
        "Book toppling off the shelf"
    ],
    "159": [
        "Tipping a cup with water over",
        "Pouring liquid out of a bottle",
        "Emptying a bag of chips onto a plate",
        "Knocking a jar of marbles off the table"
    ],
    "160": [
        "Pressing a button",
        "Rubbing a surface",
        "Gently tapping a screen",
        "Caressing a fabric"
    ],
    "161": [
        "attaching tape to wall",
        "tape not sticking",
        "trying to adhere object",
        "unsuccessful attachment"
    ],
    "162": [
        "Struggling with a rigid object",
        "Exerting force without result",
        "Attempting to bend an immovable object",
        "Failing to make any progress"
    ],
    "163": [
        "Pouring liquid",
        "Missing the target",
        "Spilling liquid",
        "Overflowing liquid"
    ],
    "164": [
        "object",
        "hand",
        "rotation",
        "inversion"
    ],
    "165": [
        "Camera pointing downwards",
        "Filming a lower perspective",
        "Capturing the ground level view",
        "Adjusting camera angle for a lower shot"
    ],
    "166": [
        "Leftward camera movement",
        "Pan to the left",
        "Filming while turning left",
        "Horizontal camera rotation to the left"
    ],
    "167": [
        "Camera turning right",
        "Filming action",
        "Directional movement",
        "Camera panning"
    ],
    "168": [
        "tilted camera angle",
        "upward camera movement",
        "pointing camera upwards",
        "filming from a higher perspective"
    ],
    "169": [
        "hands",
        "wet object",
        "twisting motion",
        "water droplets"
    ],
    "170": [
        "Knob",
        "Handle",
        "Screw",
        "Cap"
    ],
    "171": [
        "Removing a cloth",
        "Unveiling a hidden object",
        "Peeling off a cover",
        "Revealing a secret"
    ],
    "172": [
        "Unfolding paper",
        "Unfolding map",
        "Unfolding a letter",
        "Unfolding a brochure"
    ],
    "173": [
        "Cleaning",
        "Removing",
        "Erasing",
        "Removing dust"
    ]
}
# Extracting sentences from JSON and joining them into a single string
sentences = []
for key in json_data:
    sentences.extend(json_data[key])
text_content = "\n".join(sentences)

# Writing the sentences to a text file
file_path = "./ssv2_ost_spatio_concepts.txt"
with open(file_path, "w") as file:
    file.write(text_content)
