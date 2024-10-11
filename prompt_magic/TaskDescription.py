List_of_objects = []
List_of_predicates = [] 

# Example-1: [cat-1, chasing, mouse-2]
#     - cat is a subject and #id 1 is assigned to cat(subject).
#     - chasing is a predicate/action which cat-1 is performing
#     - mouse-2 is an object and #id 2 is assigned mouse(object) which is affected by cat-1(subject) and chasing(predicate)
# Example-2 [cat-4, standing next, cat-2]
#     - cat-4 is a subject and #id 4 is assigned to cat(subject).
#     - standing next is a sptial predicate which describes cat-4(subject) in the scene
#     - cat-2 is an object and #id 2 is assigned cat(object) which is affected by cat-4(subject) and standing next(predicate)

# Task: Generate scene graph triplets from the given video in the format [subject-#id, predicate, object-#id].

#
# 
# 

opvsg_predicates_numbered = """1.beside 2.biting 3.blowing 4.brushing 5.caressing 6.carrying 7.catching 8.chasing 9.cleaning 10.closing 11.cooking 12.cutting 13.drinking from 14.eating 15.entering 16.feeding 17.grabbing 18.guiding 19.hanging from 20.hitting 21.holding 22.hugging 23.in 24.in front of 25.jumping from 26.jumping over 27.kicking 28.kissing 29.licking 30.lighting 31.looking at 32.lying on 33.next to 34.on 35.opening 36.over 37.picking 38.playing 39.playing with 40.pointing to 41.pulling 42.pushing 43.riding 44.running on 45.shaking hand with 46.sitting on 47.standing on 48.stepping on 49.stirring 50.swinging 51.talking to 52.throwing 53.touching 54.toward 55.walking on 56.watering 57.wearing"""
objects_numbered_pvsg = """1.adult 2.baby 3.bag 4.ball 5.ballon 6.basket 7.bat 8.bed 9.bench 10.beverage 11.bike 12.bird 13.blanket 14.board 15.book 16.bottle 17.bowl 18.box 19.bread 20.brush 21.bucket 22.cabinet 23.cake 24.camera 25.can 26.candle 27.car 28.card 29.carpet 30.cart 31.cat 32.cellphone 33.chair 34.child 35.chopstick 36.cloth 37.computer 38.condiment 39.cookie 40.countertop 41.cover 42.cup 43.curtain 44.dog 45.door 46.drawer 47.dustbin 48.egg 49.fan 50.faucet 51.fence 52.flower 53.fork 54.fridge 55.fruit 56.gift 57.glass 58.glasses 59.glove 60.grain 61.guitar 62.hat 63.helmet 64.horse 65.iron 66.knife 67.light 68.lighter 69.mat 70.meat 71.microphone 72.microwave 73.mop 74.net 75.noodle 76.others 77.oven 78.pan 79.paper 80.piano 81.pillow 82.pizza 83.plant 84.plate 85.pot 86.powder 87.rack 88.racket 89.rag 90.ring 91.scissor 92.shelf 93.shoe 94.simmering 95.sink 96.slide 97.sofa 98.spatula 99.sponge 100.spoon 101.spray 102.stairs 103.stand 104.stove 105.switch 106.table 107.teapot 108.towel 109.toy 110.tray 111.tv 112.vaccum 113.vegetable 114.washer 115.window 116.ceiling 117.floor 118.grass 119.ground 120.rock 121.sand 122.sky 123.snow 124.tree 125.wall 126.water"""
Task_description_v10_pvsg = f"""The predefined objects_entity lexicon containing 126 lexemes is numbered as follows: """ + objects_numbered_pvsg + """
    and predefined relations_entity lexicon containing 57 lexemes is numbered as follows: """ +  opvsg_predicates_numbered + """

    Given the objects and relations lexeme, the task is to describe the provided video in deatail and create triplets in the form of [objects_entity-id lexicon, relations_entity lexicon, objects_entity-id lexicon] using the predefined entity lexicon. 
    The id is randomly assigned to each object-entity to ensure uniqueness and tracking throughout the video.

    Please note: It is possible that the exact relations_entity or objects_entity might not be visible in the video, but those can be aligned, refer below examples.
        Example-1: The relations_entity 'snuggling' can be mapped to '21.hugging'.
        Example-2: The objects_entity 'infant' can be mapped to '1.baby'.
        Example-3: The relations_entity 'transporting' can be mapped to '5.carrying'.
        Example-4: The relations_entity 'smacking' can be mapped to '19.hitting'.
        Example-5: The objects_entity 'drinks' can be mapped to '9.beverage'.
        Example-6: The objects_entity "television" can be mapped to "110.tv".


    In-context example 1:

        #sg_start
        {
            "scene": { 
                "description" : "A child and a dog play outside, with the child holding a ball while the dog jumps beside them on the grass.",
            },
            "frame-1": {
                "descriptions": ["The child holds the ball while the dog jumps beside them.", "The dog is next to the child on the grass."],
                "triplets": [["33.child-7", "20.holding", "3.ball-2"], ["43.dog-3", "24.jumping from", "33.child-7"], ["43.dog-3", "0.beside", "33.child-7"]]
            },  
            "frame-2": {
                "descriptions": ["The child throws the ball to the dog.", "The dog catches the ball mid-air."],
                "triplets": [["33.child-7", "51.throwing", "3.ball-2"], ["43.dog-3", "6.catching", "3.ball-2"]]
            },  
            "frame-3": {
                "descriptions": ["The child chases the dog.", "The dog runs away with the ball."],
                "triplets": [["33.child-7", "7.chasing", "43.dog-3"], ["43.dog-3", "43.running on", "117.grass-5"]]
            },
            "frame-4": {
                "descriptions": ["The child picks up a stick to throw it to the dog.", "The dog watches intently."],
                "triplets": [["33.child-7", "36.picking", "118.ground-9"], ["43.dog-3", "30.looking at", "33.child-7"]]
            },
            "frame-5": {
                "descriptions": ["The dog jumps over the child.", "The child laughs, watching the dog."],
                "triplets": [["43.dog-3", "25.jumping over", "33.child-7"], ["33.child-7", "30.looking at", "43.dog-3"]]
            },
            "frame-6": {
                "descriptions": ["The child and the dog sit on the grass together.", "They look at each other contentedly."],
                "triplets": [["33.child-7", "45.sitting on", "117.grass-5"], ["43.dog-3", "45.sitting on", "117.grass-5"], ["33.child-7", "50.talking to", "43.dog-3"]]
            },
            "frame-7": {
                "descriptions": ["The child and dog lie down together on the grass.", "The dog lies beside the child."],
                "triplets": [["33.child-7", "31.lying on", "117.grass-5"], ["43.dog-3", "31.lying on", "117.grass-5"], ["43.dog-3", "32.next to", "33.child-7"]]
            },
            "frame-8": {
                "descriptions": ["The child and dog rest quietly, side by side.", "They both look up at the sky."],
                "triplets": [["33.child-7", "0.beside", "43.dog-3"], ["43.dog-3", "30.looking at", "121.sky-4"], ["33.child-7", "30.looking at", "121.sky-4"]]
            },
            "st progression": "The scene depicts a playful interaction between a child and a dog. The child initially holds a ball, and the dog eagerly jumps around. As they play, the child throws the ball, and the dog catches it. They continue playing and eventually rest together on the grass, enjoying each other's company."
        }
        #sg_end

    In-context example 2:

        #sg_start
        {
            "scene": { 
                "description" : "An adult and a baby are in the kitchen, with the adult holding a bottle while feeding the baby who is sitting on a chair.",
            },
            "frame-1": {
                "descriptions": ["The adult holds the bottle while feeding the baby.", "The baby is sitting on the chair in front of the adult."],
                "triplets": [ ["0.adult-2", "20.holding", "15.bottle-4"], ["0.adult-2", "15.feeding", "1.baby-6"], ["1.baby-6", "45.sitting on", "32.chair"], ["1.baby-6", "23.in front of", "0.adult-2"]]
            },  
            "frame-2": {
                "descriptions": ["The baby drinks from the bottle.", "The adult looks at the baby while feeding."],
                "triplets": [["1.baby-6", "12.drinking from", "15.bottle-4"], ["0.adult-2", "30.looking at", "1.baby-6"]]
            },  
            "frame-3": {
                "descriptions": ["The baby holds onto the bottle while drinking.", "The adult supports the bottle in the baby's hands."],
                "triplets": [["1.baby-6", "20.holding", "15.bottle-4"], ["0.adult-2", "53.toward", "1.baby-6"]]
            },
            "frame-4": {
                "descriptions": ["The adult gently caresses the baby's head.", "The baby continues to drink from the bottle."],
                "triplets": [["0.adult-2", "4.caressing", "1.baby-6"], ["1.baby-6", "12.drinking from", "15.bottle-4"]]
            },
            "frame-5": {
                "descriptions": ["The baby smiles up at the adult.", "The adult smiles back, holding the bottle."],
                "triplets": [["1.baby-6", "30.looking at", "0.adult-2"], ["0.adult-2", "30.looking at", "1.baby-6"], ["0.adult-2", "20.holding", "15.bottle-4"]]
            },
            "frame-6": {
                "descriptions": ["The adult places the baby on the ground.", "The baby crawls next to the adult."],
                "triplets": [["0.adult-2", "47.stepping on", "116.floor-0"], ["1.baby-6", "31.lying on", "116.floor-0"], ["1.baby-6", "0.beside", "0.adult-2"]]
            },
            "frame-7": {
                "descriptions": ["The adult picks up a toy for the baby.", "The baby watches, curious."],
                "triplets": [["0.adult-2", "36.picking", "108.toy-9"], ["1.baby-6", "30.looking at", "0.adult-2"]]
            },
            "frame-8": {
                "descriptions": ["The adult and baby play with the toy together.", "They are both sitting on the floor, laughing."],
                "triplets": [["0.adult-2", "38.playing with", "108.toy-9"], ["1.baby-6", "38.playing with", "108.toy-9"], ["0.adult-2", "45.sitting on", "116.floor-0"], ["1.baby-6", "45.sitting on", "116.floor-0"]]
            },
            "st progression": "The scene portrays a nurturing interaction between an adult and a baby. The adult holds a bottle and feeds the baby, while the baby engages by drinking and eventually holding the bottle. Afterward, they play together with a toy, sharing smiles and laughter."
        }
        #sg_end
    
    
    Now, from the provided video which has 8 frames generate the triplets, output format should be same as above In-context examples: 
"""


predicates_numbered = """1.jump right 2.stand left 3.taller 4.jump past 5.jump behind 6.stand front 7.sit next to 8.sit behind 9.sit front 10.next to 11.front 12.stand next to 13.stand behind 14.walk right 15.walk next to 16.walk left 17.walk past 18.walk front 19.walk behind 20.faster 21.larger 22.stand with 23.stand right 24.walk with 25.walk toward 26.walk away 27.stop right 28.stop beneath 29.stand above 30.ride 31.run beneath 32.sit above 33.sit beneath 34.sit left 35.sit right 36.walk above 37.behind 38.watch 39.hold 40.feed 41.touch 42.right 43.left 44.follow 45.move front 46.move beneath 47.chase 48.run left 49.run right 50.lie next to 51.lie behind 52.play 53.move behind 54.jump beneath 55.fly with 56.fly past 57.move right 58.move left 59.swim front 60.swim left 61.move with 62.jump front 63.jump left 64.swim right 65.swim next to 66.jump next to 67.swim with 68.move past 69.bite 70.pull 71.jump toward 72.fight 73.run front 74.run behind 75.sit inside 76.drive 77.lie front 78.stop behind 79.lie left 80.stop left 81.lie right 82.creep behind 83.creep above 84.beneath 85.above 86.fall off 87.stop front 88.run away 89.run next to 90.away 91.jump away 92.fly next to 93.lie beneath 94.jump above 95.lie above 96.walk beneath 97.stand beneath 98.move toward 99.toward 100.past 101.move away 102.run past 103.fly behind 104.fly above 105.fly left 106.lie with 107.creep away 108.creep left 109.creep front 110.run with 111.run toward 112.creep right 113.creep past 114.fly front 115.fly right 116.fly away 117.fly toward 118.stop above 119.stand inside 120.kick 121.run above 122.swim beneath 123.jump with 124.lie inside 125.move above 126.move next to 127.creep next to 128.creep beneath 129.swim behind 130.stop next to 131.stop with 132.creep toward"""
objects_numbered = """1.antelope 2.person 3.dog 4.zebra 5.bicycle 6.horse 7.monkey 8.fox 9.elephant 10.lion 11.giant_panda 12.airplane 13.whale 14.watercraft 15.car 16.bird 17.cattle 18.rabbit 19.snake 20.frisbee 21.motorcycle 22.ball 23.domestic_cat 24.bear 25.red_panda 26.lizard 27.skateboard 28.sheep 29.squirrel 30.bus 31.sofa 32.train 33.turtle 34.tiger 35.hamster"""
Task_description_v10 = f"""The predefined objects_entity lexicon containing 35 lexemes is numbered as follows: """ + objects_numbered + """ \n\
    and predefined relations_entity lexicon containing 132 lexemes is numbered as follows: """ + predicates_numbered + """ \n\
    
    Given the objects and relations lexeme, the task is to generate triplets from the video in the form of [objects_entity-id lexicon, relations_entity lexicon, objects_entity-id lexicon] using the predefined entity lexicon. 
    The id is randomly assigned to each object-entity to ensure uniqueness and tracking throughout the video.

    Note: It is possible that exact relations_entity or objects_entity might not be visible in the video, but those can be aligned, refer below examples.
        example-1: the relations_entity 'being touched and leaned over by' can be mapped to '41.touch'.
        example-2: the relations_entity 'sitting straighter' can be mapped to '75.sit'.
        example-3: the objects_entity 'rider' can be mapped to '2.person'.
        example-4: the objects_entity 'deer' can be mapped to '1.antelope'.

    In-context Example 1:
        #sg_start
        {
            "scene": { 
                "description" : "Two lions move through the savanna, with the larger lion leading and the smaller lion following behind. Their positions emphasize their spatial relationship and size difference.",
            },
            "f1": {
                "descriptions": ["The larger lion walks in front of the smaller lion.","The smaller lion stands behind the larger lion."],
                "triplets": [["10.lion-0", "18.walk front", "10.lion-1"], ["10.lion-1", "13.stand behind", "10.lion-0"]]
            },  
            "f2": {
                "descriptions": ["The larger lion continues walking ahead.","The smaller lion walks behind, keeping pace."],
                "triplets": [["10.lion-0", "18.walk front", "10.lion-1"], ["10.lion-1", "19.walk behind", "10.lion-0"]]
            },  
            "f3": {
                "descriptions": ["The larger lion is taller than the smaller lion.","The smaller lion walks behind, following closely."],
                "triplets": [["10.lion-0", "3.taller", "10.lion-1"], ["10.lion-1", "19.walk behind", "10.lion-0"]]
            },
            "f4": {
                "descriptions": ["The larger lion walks right while the smaller lion follows behind.","The size difference between the lions remains visible."],
                "triplets": [["10.lion-0", "14.walk right", "10.lion-1"], ["10.lion-1", "19.walk behind", "10.lion-0"]]
            },
            "f5": {
                "descriptions": ["The larger lion walks toward the front, leading the way.","The smaller lion follows behind, staying close."],
                "triplets": [["10.lion-0", "25.walk toward", "10.lion-1"], ["10.lion-1", "19.walk behind", "10.lion-0"]]
            },
            "f6": {
                "descriptions": ["The larger lion walks next to the smaller lion for a moment.","The lions maintain their positions."],
                "triplets": [["10.lion-0", "15.walk next to", "10.lion-1"], ["10.lion-1", "10.next to", "10.lion-0"]]
            },
            "f7": {
                "descriptions": ["The larger lion walks past the smaller lion.","The smaller lion stands behind, watching."],
                "triplets": [["10.lion-0", "17.walk past", "10.lion-1"], ["10.lion-1", "13.stand behind", "10.lion-0"]]
            },
            "f8": {
                "descriptions": ["The lions stop moving, the larger one still in front of the smaller lion.","The smaller lion sits behind the larger one."],
                "triplets": [["10.lion-0", "87.stop front", "10.lion-1"], ["10.lion-1", "8.sit behind", "10.lion-0"]]
            },
            "st progression": "The scene shows the movement and spatial relationship of the two lions in the savanna. The larger lion consistently leads in size and position, with the smaller lion following closely behind or beside, eventually stopping behind the larger lion."
        }
        #sg_end

    In-context Example 2:
        #sg_start
        {
            "scene": { 
                "description" : "Two bicycles and two people are interacting as they move through the scene, with the people riding and maneuvering the bicycles. The scene highlights their movements and spatial relationships.",
            },
            "f1": {
                "descriptions": ["The first bicycle moves right of the second bicycle.","The first bicycle moves in front of the second bicycle."],
                "triplets": [["5.bicycle-0", "57.move right", "5.bicycle-1"], ["5.bicycle-0", "45.move front", "5.bicycle-1"]]
            },  
            "f2": {
                "descriptions": ["Both bicycles move together side by side.","The second bicycle follows the first."],
                "triplets": [["5.bicycle-0", "61.move with", "5.bicycle-1"], ["5.bicycle-1", "61.move with", "5.bicycle-0"]]
            },  
            "f3": {
                "descriptions": ["The second bicycle moves behind the first.","The second bicycle shifts to the left of the first."],
                "triplets": [["5.bicycle-1", "19.move behind", "5.bicycle-0"], ["5.bicycle-1", "58.move left", "5.bicycle-0"]]
            },
            "f4": {
                "descriptions": ["The first bicycle moves beneath the person.","The person sits above the first bicycle, riding it."],
                "triplets": [["5.bicycle-0", "46.move beneath", "2.person-3"], ["2.person-3", "32.sit above", "5.bicycle-0"]]
            },
            "f5": {
                "descriptions": ["The first bicycle moves right of another person.","The first bicycle moves in front of the other person."],
                "triplets": [["5.bicycle-0", "57.move right", "2.person-5"], ["5.bicycle-0", "45.move front", "2.person-5"]]
            },
            "f6": {
                "descriptions": ["The other person stands behind the first bicycle.","The other person stands left of the first bicycle."],
                "triplets": [["2.person-5", "37.behind", "5.bicycle-0"], ["2.person-5", "44.left", "5.bicycle-0"]]
            },
            "f7": {
                "descriptions": ["The second bicycle moves behind the person.","The second bicycle moves left of the person."],
                "triplets": [["5.bicycle-1", "19.move behind", "2.person-3"], ["5.bicycle-1", "58.move left", "2.person-3"]]
            },
            "f8": {
                "descriptions": ["The person sits above the second bicycle and rides it.","The two people align with each other, moving through the scene."],
                "triplets": [["2.person-5", "32.sit above", "5.bicycle-1"], ["2.person-5", "61.move with", "2.person-3"]]
            },
            "st progression": "The two bicycles and two people move through the scene, interacting closely. The bicycles are ridden by the people, with movements emphasizing spatial relationships such as moving left, right, in front, and behind."
        }
        #sg_end

    Now, from the provided video generate the triplets. Answer: 
"""



Task_description_v9 = """Identify and describe the objects in this video scene, their spatial positions relative to each other, and the actions they are performing over time. 
Focus on extracting spatio-temporal relationships, and for each frame, provide triplets of the form [Subject, Relation, Object] where you describe the position, action, or change between objects. 
Also, describe how these interactions evolve as time progresses. 

    Include the following details:
        - Subjects/Objects: Entities involved in the scene.
        - Relation:
            - Relations: Spatial or temporal interactions between the objects (e.g., 'next to', 'on top of', 'holding', 'stand behind', 'walk next to' etc.).
            - Actions: Movements or actions over time (e.g., 'walking', 'falling', 'picking up', etc.).
            - Temporal Changes: How the relationships or positions change across different frames.

    Other important things to consider:
        - You are provided with specific lists of Relations/Actions/Temporal Changes,and you must only consider those when generating scene graph triplets.
        - Do not use any relations if it does not accurately describe an action or spatial relationship between the objects present in the video.
    
    In-context Example 1:
        Given list of Relations=[jump front,sitting on,walk behind,standing next to,stand front,stand behind,walk front,reaching for,catching,holding,chasing,walking toward]

        #sg_start
        {
            "scene": {
                "description": "A person is sitting on a chair, picks up a book from the table, and then walks toward the door."
            },
            "f1": {
                "descriptions": ["The person is sitting on a chair.", "The table is positioned next to the person."],
                "triplets": [["person-7", "sitting on", "chair-9"], ["table-5", "next to", "person-7"]]
            },  
            "f2": {
                "descriptions": ["The person is reaching out to pick up the book from the table."],
                "triplets": [["person-7", "reaching for", "book-4"]]
            },  
            "f3": {
                "descriptions": ["The person picks up the book and is now holding it."],
                "triplets": [["person-7", "holding", "book-4"]]
            },
            "f4": {
                "descriptions": ["The person walks away from the chair toward the door."],
                "triplets": [["person-7", "walking toward", "door-0"]]
            },
            "st progression": "The person begins by sitting on the chair. After picking up the book from the table, they change their position by walking toward the door"
        }
        #sg_end

    In-context Example 2:
        Given list of Relations=[jump front,running in,jumping over,walk behind,in front of,stand front,stand behind,walk front,reaching for,catching,holding,chasing,walking toward]

        #sg_start
        {
            "scene": { 
                "description" : "A dog runs in a park, jumps over a fence, and chases a ball.",
            },
            "f1": {
                "descriptions": ["The dog is running in the park.","A fence is positioned in front of the dog."],
                "triplets": [["dog-2", "running in", "park-4"], ["fence-2", "in front of", "dog-2"]]
            },  
            "f2": {
                "descriptions": ["The dog jumps over the fence."],
                "triplets": [["dog-2", "jumping over", "fence-2"]]
            },  
            "f3": {
                "descriptions": ["The dog starts chasing a ball on the other side of the fence."],
                "triplets": [["dog-2", "chasing", "ball-3"]]
            },
            "f4": {
                "descriptions": ["The dog catches up to the ball."],
                "triplets": [["dog-2", "reaching", "ball-3"]]
            },
            "st progression": "The dog begins by running through the park, encountering a fence. The spatial relation changes when the dog jumps over the fence, and the temporal relation emerges as the dog begins chasing and eventually reaches the ball."
        }
        #sg_end

    Note that only given list of relations are used to describe video and generate triplets
    Now given list of Relations=[{List_of_predicates}]
"""



Task_description_v8 = """Identify and describe the objects in this video scene, their spatial positions relative to each other, and the actions they are performing over time. 
Focus on extracting spatio-temporal relationships, and for each frame, provide triplets of the form [Subject, Relation, Object] where you describe the position, action, or change between objects. 
Also, describe how these interactions evolve as time progresses. 

    Include the following details:
        - Subjects/Objects: Entities involved in the scene.
        - Relation:
            - Relations: Spatial or temporal interactions between the objects (e.g., 'next to', 'on top of', 'holding', 'stand behind', 'walk next to' etc.).
            - Actions: Movements or actions over time (e.g., 'walking', 'falling', 'picking up', etc.).
            - Temporal Changes: How the relationships or positions change across different frames.

    Other important things to consider:
        - You are provided with specific lists of Relations/Actions/Temporal Changes,and you must only consider those when generating scene graph triplets.
        - Do not use any Relations/Actions/Temporal Changes if it does not describe an action or spatial relationship between the objects present in the video.
        - If Subjects/Objects from the provided list does not appear in the video, do not generate triplet for using such Subjects/Objects.
    
    In-context Example 1:
        Given list of Objects=[mountain, river,person, book, door,horse, train, plane,chair,monkey] and 
        Relations/Actions/Temporal Changes=[jump front,sitting on,walk behind,standing next to,stand front,stand behind,walk front,reaching for,catching,holding,chasing,walking toward]

        #sg_start
        {
            "scene": {
                "description": "A person is sitting on a chair, picks up a book from the table, and then walks toward the door."
            },
            "f1": {
                "descriptions": ["The person is sitting on a chair.", "The table is positioned next to the person."],
                "triplets": [["person-7", "sitting on", "chair-9"], ["table-5", "next to", "person-7"]]
            },  
            "f2": {
                "descriptions": ["The person is reaching out to pick up the book from the table."],
                "triplets": [["person-7", "reaching for", "book-4"]]
            },  
            "f3": {
                "descriptions": ["The person picks up the book and is now holding it."],
                "triplets": [["person-7", "holding", "book-4"]]
            },
            "f4": {
                "descriptions": ["The person walks away from the chair toward the door."],
                "triplets": [["person-7", "walking toward", "door-0"]]
            },
            "st progression": "The person begins by sitting on the chair. After picking up the book from the table, they change their position by walking toward the door"
        }
        #sg_end

    In-context Example 2:
        Given list of Objects=[fence,mountain,river,person, book,ball,door,horse,park,train, plane,chair,monkey, dog] and 
        Relations/Actions/Temporal Changes=[jump front,running in,jumping over,walk behind,in front of,stand front,stand behind,walk front,reaching for,catching,holding,chasing,walking toward]

        #sg_start
        {
            "scene": { 
                "description" : "A dog runs in a park, jumps over a fence, and chases a ball.",
            },
            "f1": {
                "descriptions": ["The dog is running in the park.","A fence is positioned in front of the dog."],
                "triplets": [["dog-2", "running in", "park-4"], ["fence-2", "in front of", "dog-2"]]
            },  
            "f2": {
                "descriptions": ["The dog jumps over the fence."],
                "triplets": [["dog-2", "jumping over", "fence-2"]]
            },  
            "f3": {
                "descriptions": ["The dog starts chasing a ball on the other side of the fence."],
                "triplets": [["dog-2", "chasing", "ball-3"]]
            },
            "f4": {
                "descriptions": ["The dog catches up to the ball."],
                "triplets": [["dog-2", "reaching", "ball-3"]]
            },
            "st progression": "The dog begins by running through the park, encountering a fence. The spatial relation changes when the dog jumps over the fence, and the temporal relation emerges as the dog begins chasing and eventually reaches the ball."
        }
        #sg_end

    Note that only given list of relations are used to describe video and generate triplets
    Now given list of Objects=[{List_of_objects}] and 
    Relations=[{List_of_predicates}]
"""

Task_description_v7 = """Identify and describe the objects in this video scene, their spatial positions relative to each other, and the actions they are performing over time. 
Focus on extracting spatio-temporal relationships, and for each frame, provide triplets of the form [Subject, Relation, Object] where you describe the position, action, or change between objects. 
Also, describe how these interactions evolve as time progresses. 

    Include the following details:
        - Subjects/Objects: Entities involved in the scene.
        - Relation:
            - Relations: Spatial or temporal interactions between the objects (e.g., 'next to', 'on top of', 'holding', 'stand behind', 'walk next to' etc.).
            - Actions: Movements or actions over time (e.g., 'walking', 'falling', 'picking up', etc.).
            - Temporal Changes: How the relationships or positions change across different frames.

    Other important things to consider:
        - You are provided with specific lists of Relations/Actions/Temporal Changes,and you must only consider those when generating scene graph triplets.
        - Do not use any relations if it does not accurately describe an action or spatial relationship between the objects present in the video.
    
    In-context Example 1:
        Given list of Relations=[jump front,sitting on,walk behind,standing next to,stand front,stand behind,walk front,reaching for,catching,holding,chasing,walking toward]

        #sg_start
        {
            "scene": {
                "description": "A person is sitting on a chair, picks up a book from the table, and then walks toward the door."
            },
            "f1": {
                "descriptions": ["The person is sitting on a chair.", "The table is positioned next to the person."],
                "triplets": [["person-7", "sitting on", "chair-9"], ["table-5", "next to", "person-7"]]
            },  
            "f2": {
                "descriptions": ["The person is reaching out to pick up the book from the table."],
                "triplets": [["person-7", "reaching for", "book-4"]]
            },  
            "f3": {
                "descriptions": ["The person picks up the book and is now holding it."],
                "triplets": [["person-7", "holding", "book-4"]]
            },
            "f4": {
                "descriptions": ["The person walks away from the chair toward the door."],
                "triplets": [["person-7", "walking toward", "door-0"]]
            },
            "st progression": "The person begins by sitting on the chair. After picking up the book from the table, they change their position by walking toward the door"
        }
        #sg_end

    In-context Example 2:
        Given list of Relations=[jump front,running in,jumping over,walk behind,in front of,stand front,stand behind,walk front,reaching for,catching,holding,chasing,walking toward]

        #sg_start
        {
            "scene": { 
                "description" : "A dog runs in a park, jumps over a fence, and chases a ball.",
            },
            "f1": {
                "descriptions": ["The dog is running in the park.","A fence is positioned in front of the dog."],
                "triplets": [["dog-2", "running in", "park-4"], ["fence-2", "in front of", "dog-2"]]
            },  
            "f2": {
                "descriptions": ["The dog jumps over the fence."],
                "triplets": [["dog-2", "jumping over", "fence-2"]]
            },  
            "f3": {
                "descriptions": ["The dog starts chasing a ball on the other side of the fence."],
                "triplets": [["dog-2", "chasing", "ball-3"]]
            },
            "f4": {
                "descriptions": ["The dog catches up to the ball."],
                "triplets": [["dog-2", "reaching", "ball-3"]]
            },
            "st progression": "The dog begins by running through the park, encountering a fence. The spatial relation changes when the dog jumps over the fence, and the temporal relation emerges as the dog begins chasing and eventually reaches the ball."
        }
        #sg_end

    Note that only given list of relations are used to describe video and generate triplets
    Now given list of Relations=[{List_of_predicates}]
"""


# For example: 
# if subjects/objects: [person, cup, child, sofa, chair, table, cake] and predicates: [running on, sitting on, holding, picking, jumping] then based on the scene triplets can be #sg_start [person-8,holding,cup-2];[child-3,running on,floor-0];[child-6,sitting on,sofa-10] #sg_end  

Task_description_v6 = """Identify and describe the objects in this video scene, their spatial positions relative to each other, and the actions they are performing over time. 
Focus on extracting spatio-temporal relationships, and for each frame, provide triplets of the form [Subject, Relation, Object] where you describe the position, action, or change between objects. 
Also, describe how these interactions evolve as time progresses. 

    Include the following details:
        - Objects: Entities involved in the scene.
        - Relations: Spatial or temporal interactions between the objects (e.g., 'next to', 'on top of', 'holding', 'stand behind', 'walk next to' etc.).
        - Actions: Movements or actions over time (e.g., 'walking', 'falling', 'picking up', etc.).
        - Temporal Changes: How the relationships or positions change across different frames.
    
    In-context Example 1:
        #sg_start
        {
            "Scene": "A person is sitting on a chair, picks up a book from the table, and then walks toward the door.",
            "F1": {
                "descriptions": ["The person is sitting on a chair.", "The table is positioned next to the person."],
                "triplets": [["person-7", "sitting on", "Chair-9"], ["Table-5", "next to", "person-7"]]
            },  
            "F2": {
                "descriptions": ["The person is reaching out to pick up the book from the table."],
                "triplets": [["person-7", "reaching for", "Book-4"]]
            },  
            "F3": {
                "descriptions": ["The person picks up the book and is now holding it."],
                "triplets": [["person-7", "holding", "Book-4"]]
            },
            "F4": {
                "descriptions": ["The person walks away from the chair toward the door."],
                "triplets": [["person-7", "walking toward", "Door-0"]]
            },
            "ST Progression": "The person begins by sitting on the chair. After picking up the book from the table, they change their position by walking toward the door"
        }
        #sg_end

    In-context Example 2:

        #sg_start
        {
            "Scene" : "A dog runs in a park, jumps over a fence, and chases a ball.",
            "F1": {
                "descriptions": ["The dog is running in the park.","A fence is positioned in front of the dog."],
                "triplets": [["dog-2", "running in", "Park-4"], ["fence-2", "in front of", "dog-2"]]
            },  
            "F2": {
                "descriptions": ["The dog jumps over the fence."],
                "triplets": [["dog-2", "jumping over", "fence-2"]]
            },  
            "F3": {
                "descriptions": ["The dog starts chasing a ball on the other side of the fence."],
                "triplets": [["dog-2", "chasing", "ball-3"]]
            },
            "F4": {
                "descriptions": ["The dog catches up to the ball."],
                "triplets": [["dog-2", "reaching", "ball-3"]]
            },
            "ST Progression": "The dog begins by running through the park, encountering a fence. The spatial relation changes when the dog jumps over the fence, and the temporal relation emerges as the dog begins chasing and eventually reaches the ball."
        }
        #sg_end
"""



Task_description_v5 = """
Task: Describe provided video in detailed manner in the triplets format [subject-#id, predicate, object-#id] using provided list of subjects/objects and predicates.

- subject is an entity who is performing action in the scene, or entity selected to describe it's spatial placement in the scene (e.g. antelope, table, cat, person etc.)
- The #id in (subject-#id and object-#id) is a randomly assigned unique identifier used to distinguish objects of the same category and track the objects throughout the scene.
- predicate describes what subject is doing in the scene with object or describes subject's placement in the scene with respect to other objects. (e.g. holding, standing next to, drinking etc.)
- object is an entity which receives the action done by subject or its used as a reference to which subjects spatial position is given.


In-context Example 1:
    #sg_start
    {
        "Scene": "A person is sitting on a chair, picks up a book from the table, and then walks toward the door.",
        "F1": {
            "descriptions": ["The person is sitting on a chair.", "The table is positioned next to the person."],
            "triplets": [["person-7", "sitting on", "Chair-9"], ["Table-5", "next to", "person-7"]]
        },  
        "F2": {
            "descriptions": ["The person is reaching out to pick up the book from the table."],
            "triplets": [["person-7", "reaching for", "Book-4"]]
        },  
        "F3": {
            "descriptions": ["The person picks up the book and is now holding it."],
            "triplets": [["person-7", "holding", "Book-4"]]
        },
        "F4": {
            "descriptions": ["The person walks away from the chair toward the door."],
            "triplets": [["person-7", "walking toward", "Door-0"]]
        },
        "ST Progression": "The person begins by sitting on the chair. After picking up the book from the table, they change their position by walking toward the door"
    }
    #sg_end


In-context Example 2:
    #sg_start
    {
        "Scene" : "A dog runs in a park, jumps over a fence, and chases a ball.",
        "F1": {
            "descriptions": ["The dog is running in the park.","A fence is positioned in front of the dog."],
            "triplets": [["dog-2", "running in", "Park-4"], ["fence-2", "in front of", "dog-2"]]
        },  
        "F2": {
            "descriptions": ["The dog jumps over the fence."],
            "triplets": [["dog-2", "jumping over", "fence-2"]]
        },  
        "F3": {
            "descriptions": ["The dog starts chasing a ball on the other side of the fence."],
            "triplets": [["dog-2", "chasing", "ball-3"]]
        },
        "F4": {
            "descriptions": ["The dog catches up to the ball."],
            "triplets": [["dog-2", "reaching", "ball-3"]]
        },
        "ST Progression": "The dog begins by running through the park, encountering a fence. The spatial relation changes when the dog jumps over the fence, and the temporal relation emerges as the dog begins chasing and eventually reaches the ball."
    }
    #sg_end

Now, Describe provided video in detailed manner, Use the provided subjects/objects: {List_of_objects} and predicates: {List_of_predicates}
"""


# Task_description = """
# Task: Describe provided video in detailed manner using Scene Graph Triplets in the format [subject-#id, predicate, object-#id].

# - subject is an entity who is performing action in the scene, or entity selected to describe it's spatial placement in the scene (e.g. antelope, table, cat, person etc.)
# - The #id in (subject-#id and object-#id) is a unique identifier used to distinguish objects of the same category (e.g., adult-1, adult-3, table-4 etc.)
# - predicate describes what subject is doing in the scene with object or describes subject's placement in the scene with respect to other objects. (e.g. holding, standing next to, drinking etc.)
# - object is an entity which receives the action done by subject or its used as a reference to which subjects spatial position is given.

# For example: [person-1,holding,cup-2];[child-3,running on,floor-0];[child-3,sitting on,sofa-10]

# Now follow the steps below.
# 1. Generate triplets from the provided video in the format [subject-#id, predicate, object-#id], use the provided subjects/objects and predicate lists to generate triplets: subjects/objects: {List_of_objects} and predicates: {List_of_predicates}.
# """


# Task_description = """
#     Task: Describe video using Scene Graph Triplets in the format [subject-#id, predicate, object-#id].

#     Definitions:
#     - subject-#id: A visual object or noun in the video that performs an action or plays a role(actor) in the scene. The #id is a unique identifier used to distinguish objects of the same category (e.g., multiple people).
#     - predicate: Describes the action (verb) or spatial relationship between objects in the scene.
#     - object-#id: A visual object or noun in the video affected by the subject's action or shows spatial relationship with respect to other objects in the scene.

#     Example 1:
#         - [person-1,holding,cup-2];[child-3,running on,floor-0]
    
#     Notes: 
#         - Remember to assign unique #id to all objects without repeating the same objects in the scene.
#         - There can be N number of same category of objects but make sure they are visible in the video before assigning them the #id.

#     Now follow the steps below:

#     1. From the given video, construct meaningful scene graph triplets in the format [subject-#id, predicate, object-#id].
#         - Assign a unique #id to each object if more than one of the same category is present.
#         - Ensure that the subjects/objects in the triplets are visible in the video.
#     """

# 2. Using the output from Step 1, map each subject, object, and predicate to the corresponding entries in the provided list of subjects/objects {List_of_objects} and predicates {List_of_predicates}.
#         - Keep the #ids consistent during mapping.
#         - Ensure that the meaning of the original triplets remains unchanged during the mapping process.

# Notes:
#     - Triplets must be unique and describe meaningful interactions or spatial relations between objects in the scene.
#     - Use common sense when generating or mapping the triplets.
#     - If an object in the provided list is not visible in the video, do not use it to generate triplets.

Task_description_v3 = """
    Task: Scene graph triplet prediction in the format [subject-#id, predicate, object-#id] from the given video.
        - subject is a visual object or noun which is present in the video which can be seen as an entity who is performing action or is an actor in the scene.
            - here #id in subject-#id describes or tags the object visible in the scene to have uniqueness, if more than one objects of similar category are visible in the scene.
        - predicate can be an action(verb) which describes what the subject is doing in the scene or spatial relation for how objects are positioned to each other in the scene.
        - object is a visual object or noun which is present in the video which is described by the predicate and subject combination, this object is affected by subject and predicate.

    Notes: 
        - Remember to assign unique #id to all objects without repeating the same objects in the scene.
        - There can be N number of same category of objects but make sure they are visible in the video before assigning them the #id.

    
    Now follow the below steps,

    1. Describe the given video in the form of meaningful scene graph triplets in the format [subject-#id, predicate, object-#id]
    2. Take each subjects/objects from the step-1 and use the following list of subject/objects {List_of_objects} and predicates {List_of_predicates} and align subjects/objects without changing the #id, or predicates such that it does not change the meaning of the original triplets obtained in step-1.
   
    Notes:
        - Make sure the triplets are meaningful and with common sense and unique
        - Triplets must have subjects/objects and predicates from the provided list above and should be present in the video.
        - Make sure to cover all subject/objects to describe their spatial postions.
        - if objects don't appear in the video which is provided in the list, ignore it.
    """



    # 1. {Detect_Count_objects[v0]}
    # 2. Use provided list of objects {List_of_objects}, and if any of the detected objects in step-1 are semantically same, assign them new category from the provided list but keep the assigned #id.
    # 3. Create scene graph triplets in the format [subject-#id, predicate, object-#id].

#3. From the below provided list of objects check if objects obtained in step-2 are semantically same in the below provided list, if so use the object name from the provided list of objects

# 4. Now based on the provided video identify the relationships between objects and use objects constructed in step 3 and provided list of predicates to construct meaningful scene graph triplets the format [subject-#id, predicate, object-#id]
#         - Make sure the triplets are meaningful and unique
#         - Use objects name and #id from step-3



Task_description_v2 = """
    Your task is to generate meaningful scene graph triplets for the provided video, to do so follow the steps below.

    1. Identify objects present in the video.
    2. Assign uniqe #id to all the objects obtained in step-1 for their identity and tracking (e.g dog-1, dog-2, cat-9, mouse-8), the #id assignment should stay consistant throughout the video. The #ids count should match with total number of objects in video.
    3. Use the provided list of objects and replace if the objects are semantically same.
        - For example, if video consist of a person, and provided list has adult as an object elment then use adult as an object.
    4. For each objects from step-3 and provided predicates below, construct scene graph triplets in the format [subject-#id, predicate, object-#id] which describes the provided video.
        - Make sure the triplets are meaningful and unique
    """
Task_description_v1 = """
    Your task is to generate meaningful scene graph triplets for the provided video, to do so follow the steps below.

    1. Identify objects present in the video.
        1.1 Use below provided list of objects and use objects name from the list if they are semantically the same.
    2. Assign uniqe #id to each objects obtained in step-1 for their identity and tracking (e.g dog-1, dog-2, cat-9, mouse-8), the #id assignment should stay consistant throughout the video. The max #ids count should match with total number of objects in step-1.
    3. Using objects obtained in step-3, contruct meaningful scene graph triplets the format [subject-#id, predicate, object-#id] from the video.
        - Note: Use below provided list of predicates to construct the triplets.
    """

Task_description_v0 = """
    Your task is to generate meaningful scene graph triplets for the provided video, to do so follow the steps below.

    1. Use the provided list to identify objects in the video. Only consider objects that appear both in the video and are on the list.
    2. Use the provided list of spatial predicates and action predicates that are happening in the video between objects.
    3. Finally, construct meaningful scene graph triplets in the format [subject-#id, predicate, object-#id] with a common sense, where the subject is performing the action (predicate), and the object is receiving the action. The #id refers to the unique identifier assigned to each object in the video.
    
    Important things to consider: 
        1. Ignore any objects in the video that are not on the list, as well as any objects on the list that do not appear in the video
        2. Track the objects in the consecutive frames to reassign the same #ids to the objects.
        3. Make sure the triplets constructed are logical and with common sense.
    """