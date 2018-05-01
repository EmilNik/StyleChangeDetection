import numpy as np
import pyphen
import math
import csv
from nltk import sent_tokenize, word_tokenize, Text, pos_tag, ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from collections import Counter


with open('../data/most_common_pos_tag_trigrams.csv', 'r') as f:
    MOST_COMMON_POS_TAG_TRIGRAMS = []
    reader = csv.reader(f)
    for line in reader:
        MOST_COMMON_POS_TAG_TRIGRAMS.append(tuple(line))


# Feature extraction
class StylometryExtractor:
    DALE_CHALL_WORDS = set([
        "a", "able", "aboard", "about", "above", "absent", "accept", "accident", "account", "ache", "aching", "acorn", "acre", "across", "act", "acts", "add", "address", "admire", "adventure", "afar", "afraid", "after", "afternoon", "afterward", "afterwards", "again", "against", "age", "aged", "ago", "agree", "ah", "ahead", "aid", "aim", "air", "airfield", "airplane", "airport", "airship", "airy", "alarm", "alike", "alive", "all", "alley", "alligator", "allow", "almost", "alone", "along", "aloud", "already", "also", "always", "am", "america", "american", "among", "amount", "an", "and", "angel", "anger", "angry", "animal", "another", "answer", "ant", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anywhere", "apart", "apartment", "ape", "apiece", "appear", "apple", "april", "apron", "are", "aren't", "arise", "arithmetic", "arm", "armful", "army", "arose", "around", "arrange", "arrive", "arrived", "arrow", "art", "artist", "as", "ash", "ashes", "aside", "ask", "asleep", "at", "ate", "attack", "attend", "attention", "august", "aunt", "author", "auto", "automobile", "autumn", "avenue", "awake", "awaken", "away", "awful", "awfully", "awhile", "ax", "axe", "baa", "babe", "babies", "back", "background", "backward", "backwards", "bacon", "bad", "badge", "badly", "bag", "bake", "baker", "bakery", "baking", "ball", "balloon", "banana", "band", "bandage", "bang", "banjo", "bank", "banker", "bar", "barber", "bare", "barefoot", "barely", "bark", "barn", "barrel", "base", "baseball", "basement", "basket", "bat", "batch", "bath", "bathe", "bathing", "bathroom", "bathtub", "battle", "battleship", "bay", "be", "beach", "bead", "beam", "bean", "bear", "beard", "beast", "beat", "beating", "beautiful", "beautify", "beauty", "became", "because", "become", "becoming", "bed", "bedbug", "bedroom", "bedspread", "bedtime", "bee", "beech", "beef", "beefsteak", "beehive", "been", "beer", "beet", "before", "beg", "began", "beggar", "begged", "begin", "beginning", "begun", "behave", "behind", "being", "believe", "bell", "belong", "below", "belt", "bench", "bend", "beneath", "bent", "berries", "berry", "beside", "besides", "best", "bet", "better", "between", "bib", "bible", "bicycle", "bid", "big", "bigger", "bill", "billboard", "bin", "bind", "bird", "birth", "birthday", "biscuit", "bit", "bite", "biting", "bitter", "black", "blackberry", "blackbird", "blackboard", "blackness", "blacksmith", "blame", "blank", "blanket", "blast", "blaze", "bleed", "bless", "blessing", "blew", "blind", "blindfold", "blinds", "block", "blood", "bloom", "blossom", "blot", "blow", "blue", "blueberry", "bluebird", "bluejay", "blush", "board", "boast", "boat", "bob", "bobwhite", "bodies", "body", "boil", "boiler", "bold", "bone", "bonnet", "boo", "book", "bookcase", "bookkeeper", "boom", "boot", "born", "borrow", "boss", "both", "bother", "bottle", "bottom", "bought", "bounce", "bow", "bow-wow", "bowl", "box", "boxcar", "boxer", "boxes", "boy", "boyhood", "bracelet", "brain", "brake", "bran", "branch", "brass", "brave", "bread", "break", "breakfast", "breast", "breath", "breathe", "breeze", "brick", "bride", "bridge", "bright", "brightness", "bring", "broad", "broadcast", "broke", "broken", "brook", "broom", "brother", "brought", "brown", "brush", "bubble", "bucket", "buckle", "bud", "buffalo", "bug", "buggy", "build", "building", "built", "bulb", "bull", "bullet", "bum", "bumblebee", "bump", "bun", "bunch", "bundle", "bunny", "burn", "burst", "bury", "bus", "bush", "bushel", "business", "busy", "but", "butcher", "butt", "butter", "buttercup", "butterfly", "buttermilk", "butterscotch", "button", "buttonhole", "buy", "buzz", "by", "bye", "cab", "cabbage", "cabin", "cabinet", "cackle", "cage", "cake", "calendar", "calf", "call", "caller", "calling", "came", "camel", "camp", "campfire", "can", "can't", "canal", "canary", "candle", "candlestick", "candy", "cane", "cannon", "cannot", "canoe", "canyon", "cap", "cape", "capital", "captain", "car", "card", "cardboard", "care", "careful", "careless", "carelessness", "carload", "carpenter", "carpet", "carriage", "carrot", "carry", "cart", "carve", "case", "cash", "cashier", "castle", "cat", "catbird", "catch", "catcher", "caterpillar", "catfish", "catsup", "cattle", "caught", "cause", "cave", "ceiling", "cell", "cellar", "cent", "center", "cereal", "certain", "certainly", "chain", "chair", "chalk", "champion", "chance", "change", "chap", "charge", "charm", "chart", "chase", "chatter", "cheap", "cheat", "check", "checkers", "cheek", "cheer", "cheese", "cherry", "chest", "chew", "chick", "chicken", "chief", "child", "childhood", "children", "chill", "chilly", "chimney", "chin", "china", "chip", "chipmunk", "chocolate", "choice", "choose", "chop", "chorus", "chose", "chosen", "christen", "christmas", "church", "churn", "cigarette", "circle", "circus", "citizen", "city", "clang", "clap", "class", "classmate", "classroom", "claw", "clay", "clean", "cleaner", "clear", "clerk", "clever", "click", "cliff", "climb", "clip", "cloak", "clock", "close", "closet", "cloth", "clothes", "clothing", "cloud", "cloudy", "clover", "clown", "club", "cluck", "clump", "coach", "coal", "coast", "coat", "cob", "cobbler", "cocoa", "coconut", "cocoon", "cod", "codfish", "coffee", "coffeepot", "coin", "cold", "collar", "college", "color", "colored", "colt", "column", "comb", "come", "comfort", "comic", "coming", "company", "compare", "conductor", "cone", "connect", "coo", "cook", "cooked", "cookie", "cookies", "cooking", "cooky", "cool", "cooler", "coop", "copper", "copy", "cord", "cork", "corn", "corner", "correct", "cost", "cot", "cottage", "cotton", "couch", "cough", "could", "couldn't", "count", "counter", "country", "county", "course", "court", "cousin", "cover", "cow", "coward", "cowardly", "cowboy", "cozy", "crab", "crack", "cracker", "cradle", "cramps", "cranberry", "crank", "cranky", "crash", "crawl", "crazy", "cream", "creamy", "creek", "creep", "crept", "cried", "cries", "croak", "crook", "crooked", "crop", "cross", "cross-eyed", "crossing", "crow", "crowd", "crowded", "crown", "cruel", "crumb", "crumble", "crush", "crust", "cry", "cub", "cuff", "cuff", "cup", "cup", "cupboard", "cupful", "cure", "curl", "curly", "curtain", "curve", "cushion", "custard", "customer", "cut", "cute", "cutting", "dab", "dad", "daddy", "daily", "dairy", "daisy", "dalf", "dam", "damage", "dame", "damp", "dance", "dancer", "dancing", "dandy", "danger", "dangerous", "dare", "dark", "darkness", "darling", "darn", "dart", "dash", "date", "daughter", "dawn", "day", "daybreak", "daytime", "dead", "deaf", "deal", "dear", "death", "december", "decide", "deck", "deed", "deep", "deer", "defeat", "defend", "defense", "delight", "den", "dentist", "depend", "deposit", "describe", "desert", "deserve", "desire", "desk", "destroy", "devil", "dew", "diamond", "did", "didn't", "die", "died", "dies", "difference", "different", "dig", "dim", "dime", "dine", "ding-dong", "dinner", "dip", "direct", "direction", "dirt", "dirty", "discover", "dish", "dislike", "dismiss", "ditch", "dive", "diver", "divide", "do", "dock", "doctor", "does", "doesn't", "dog", "doll", "dollar", "dolly", "don't", "done", "donkey", "door", "doorbell", "doorknob", "doorstep", "dope", "dot", "double", "dough", "dove", "down", "downstairs", "downtown", "dozen", "drag", "drain", "drank", "draw", "draw", "drawer", "drawing", "dream", "dress", "dresser", "dressmaker", "drew", "dried", "drift", "drill", "drink", "drip", "drive", "driven", "driver", "drop", "drove", "drown", "drowsy", "drub", "drum", "drunk", "dry", "duck", "due", "dug", "dull", "dumb", "dump", "during", "dust", "dusty", "duty", "dwarf", "dwell", "dwelt", "dying", "each", "eager", "eagle", "ear", "early", "earn", "earth", "east", "eastern", "easy", "eat", "eaten", "edge", "egg", "eh", "eight", "eighteen", "eighth", "eighty", "either", "elbow", "elder", "eldest", "electric", "electricity", "elephant", "eleven", "elf", "elm", "else", "elsewhere", "empty", "end", "ending", "enemy", "engine", "engineer", "english", "enjoy", "enough", "enter", "envelope", "equal", "erase", "eraser", "errand", "escape", "eve", "even", "evening", "ever", "every", "everybody", "everyday", "everyone", "everything", "everywhere", "evil", "exact", "except", "exchange", "excited", "exciting", "excuse", "exit", "expect", "explain", "extra", "eye", "eyebrow", "fable", "face", "facing", "fact", "factory", "fail", "faint", "fair", "fairy", "faith", "fake", "fall", "false", "family", "fan", "fancy", "far", "far-off", "faraway", "fare", "farm", "farmer", "farming", "farther", "fashion", "fast", "fasten", "fat", "father", "fault", "favor", "favorite", "fear", "feast", "feather", "february", "fed", "feed", "feel", "feet", "fell", "fellow", "felt", "fence", "fever", "few", "fib", "fiddle", "field", "fife", "fifteen", "fifth", "fifty", "fig", "fight", "figure", "file", "fill", "film", "finally", "find", "fine", "finger", "finish", "fire", "firearm", "firecracker", "fireplace", "fireworks", "firing", "first", "fish", "fisherman", "fist", "fit", "fits", "five", "fix", "flag", "flake", "flame", "flap", "flash", "flashlight", "flat", "flea", "flesh", "flew", "flies", "flight", "flip", "flip-flop", "float", "flock", "flood", "floor", "flop", "flour", "flow", "flower", "flowery", "flutter", "fly", "foam", "fog", "foggy", "fold", "folks", "follow", "following", "fond", "food", "fool", "foolish", "foot", "football", "footprint", "for", "forehead", "forest", "forget", "forgive", "forgot", "forgotten", "fork", "form", "fort", "forth", "fortune", "forty", "forward", "fought", "found", "fountain", "four", "fourteen", "fourth", "fox", "frame", "free", "freedom", "freeze", "freight", "french", "fresh", "fret", "friday", "fried", "friend", "friendly", "friendship", "frighten", "frog", "from", "front", "frost", "frown", "froze", "fruit", "fry", "fudge", "fuel", "full", "fully", "fun", "funny", "fur", "furniture", "further", "fuzzy", "gain", "gallon", "gallop", "game", "gang",
        "garage", "garbage", "garden", "gas", "gasoline", "gate", "gather", "gave", "gay", "gear", "geese", "general", "gentle", "gentleman", "gentlemen", "geography", "get", "getting", "giant", "gift", "gingerbread", "girl", "give", "given", "giving", "glad", "gladly", "glance", "glass", "glasses", "gleam", "glide", "glory", "glove", "glow", "glue", "go", "goal", "goat", "gobble", "god", "god", "godmother", "goes", "going", "gold", "golden", "goldfish", "golf", "gone", "good", "good-by", "good-bye", "good-looking", "goodbye", "goodbye", "goodness", "goods", "goody", "goose", "gooseberry", "got", "govern", "government", "gown", "grab", "gracious", "grade", "grain", "grand", "grandchild", "grandchildren", "granddaughter", "grandfather", "grandma", "grandmother", "grandpa", "grandson", "grandstand", "grape", "grapefruit", "grapes", "grass", "grasshopper", "grateful", "grave", "gravel", "graveyard", "gravy", "gray", "graze", "grease", "great", "green", "greet", "grew", "grind", "groan", "grocery", "ground", "group", "grove", "grow", "guard", "guess", "guest", "guide", "gulf", "gum", "gun", "gunpowder", "guy", "ha", "habit", "had", "hadn't", "hail", "hair", "haircut", "hairpin", "half", "hall", "halt", "ham", "hammer", "hand", "handful", "handkerchief", "handle", "handwriting", "hang", "happen", "happily", "happiness", "happy", "harbor", "hard", "hardly", "hardship", "hardware", "hare", "hark", "harm", "harness", "harp", "harvest", "has", "hasn't", "haste", "hasten", "hasty", "hat", "hatch", "hatchet", "hate", "haul", "have", "haven't", "having", "hawk", "hay", "hayfield", "haystack", "he", "he'd", "he'll", "he's", "head", "headache", "heal", "health", "healthy", "heap", "hear", "heard", "hearing", "heart", "heat", "heater", "heaven", "heavy", "heel", "height", "held", "hell", "hello", "helmet", "help", "helper", "helpful", "hem", "hen", "henhouse", "her", "herd", "here", "here's", "hero", "hers", "herself", "hey", "hickory", "hid", "hidden", "hide", "high", "highway", "hill", "hillside", "hilltop", "hilly", "him", "himself", "hind", "hint", "hip", "hire", "his", "hiss", "history", "hit", "hitch", "hive", "ho", "hoe", "hog", "hold", "holder", "hole", "holiday", "hollow", "holy", "home", "homely", "homesick", "honest", "honey", "honeybee", "honeymoon", "honk", "honor", "hood", "hoof", "hook", "hoop", "hop", "hope", "hopeful", "hopeless", "horn", "horse", "horseback", "horseshoe", "hose", "hospital", "host", "hot", "hotel", "hound", "hour", "house", "housetop", "housewife", "housework", "how", "however", "howl", "hug", "huge", "hum", "humble", "hump", "hundred", "hung", "hunger", "hungry", "hunk", "hunt", "hunter", "hurrah", "hurried", "hurry", "hurt", "husband", "hush", "hut", "hymn", "i", "i'd", "i'll", "i'm", "i've", "ice", "icy", "idea", "ideal", "if", "ill", "important", "impossible", "improve", "in", "inch", "inches", "income", "indeed", "indian", "indoors", "ink", "inn", "insect", "inside", "instant", "instead", "insult", "intend", "interested", "interesting", "into", "invite", "iron", "is", "island", "isn't", "it", "it's", "its", "itself", "ivory", "ivy", "jacket", "jacks", "jail", "jam", "january", "jar", "jaw", "jay", "jelly", "jellyfish", "jerk", "jig", "job", "jockey", "join", "joke", "joking", "jolly", "journey", "joy", "joyful", "joyous", "judge", "jug", "juice", "juicy", "july", "jump", "june", "junior", "junk", "just", "keen", "keep", "kept", "kettle", "key", "kick", "kid", "kill", "killed", "kind", "kindly", "kindness", "king", "kingdom", "kiss", "kitchen", "kite", "kitten", "kitty", "knee", "kneel", "knew", "knife", "knit", "knives", "knob", "knock", "knot", "know", "known", "lace", "lad", "ladder", "ladies", "lady", "laid", "lake", "lamb", "lame", "lamp", "land", "lane", "language", "lantern", "lap", "lard", "large", "lash", "lass", "last", "late", "laugh", "laundry", "law", "lawn", "lawyer", "lay", "lazy", "lead", "leader", "leaf", "leak", "lean", "leap", "learn", "learned", "least", "leather", "leave", "leaving", "led", "left", "leg", "lemon", "lemonade", "lend", "length", "less", "lesson", "let", "let's", "letter", "letting", "lettuce", "level", "liberty", "library", "lice", "lick", "lid", "lie", "life", "lift", "light", "lightness", "lightning", "like", "likely", "liking", "lily", "limb", "lime", "limp", "line", "linen", "lion", "lip", "list", "listen", "lit", "little", "live", "lively", "liver", "lives", "living", "lizard", "load", "loaf", "loan", "loaves", "lock", "locomotive", "log", "lone", "lonely", "lonesome", "long", "look", "lookout", "loop", "loose", "lord", "lose", "loser", "loss", "lost", "lot", "loud", "love", "lovely", "lover", "low", "luck", "lucky", "lumber", "lump", "lunch", "lying", "ma", "machine", "machinery", "mad", "made", "magazine", "magic", "maid", "mail", "mailbox", "mailman", "major", "make", "making", "male", "mama", "mamma", "man", "manager", "mane", "manger", "many", "map", "maple", "marble", "march", "march", "mare", "mark", "market", "marriage", "married", "marry", "mask", "mast", "master", "mat", "match", "matter", "mattress", "may", "may", "maybe", "mayor", "maypole", "me", "meadow", "meal", "mean", "means", "meant", "measure", "meat", "medicine", "meet", "meeting", "melt", "member", "men", "mend", "meow", "merry", "mess", "message", "met", "metal", "mew", "mice", "middle", "midnight", "might", "mighty", "mile", "miler", "milk", "milkman", "mill", "million", "mind", "mine", "miner", "mint", "minute", "mirror", "mischief", "miss", "miss", "misspell", "mistake", "misty", "mitt", "mitten", "mix", "moment", "monday", "money", "monkey", "month", "moo", "moon", "moonlight", "moose", "mop", "more", "morning", "morrow", "moss", "most", "mostly", "mother", "motor", "mount", "mountain", "mouse", "mouth", "move", "movie", "movies", "moving", "mow", "mr.", "mrs.", "much", "mud", "muddy", "mug", "mule", "multiply", "murder", "music", "must", "my", "myself", "nail", "name", "nap", "napkin", "narrow", "nasty", "naughty", "navy", "near", "nearby", "nearly", "neat", "neck", "necktie", "need", "needle", "needn't", "negro", "neighbor", "neighborhood", "neither", "nerve", "nest", "net", "never", "nevermore", "new", "news", "newspaper", "next", "nibble", "nice", "nickel", "night", "nightgown", "nine", "nineteen", "ninety", "no", "nobody", "nod", "noise", "noisy", "none", "noon", "nor", "north", "northern", "nose", "not", "note", "nothing", "notice", "november", "now", "nowhere", "number", "nurse", "nut", "o'clock", "oak", "oar", "oatmeal", "oats", "obey", "ocean", "october", "odd", "of", "off", "offer", "office", "officer", "often", "oh", "oil", "old", "old-fashioned", "on", "once", "one", "onion", "only", "onward", "open", "or", "orange", "orchard", "order", "ore", "organ", "other", "otherwise", "ouch", "ought", "our", "ours", "ourselves", "out", "outdoors", "outfit", "outlaw", "outline", "outside", "outward", "oven", "over", "overalls", "overcoat", "overeat", "overhead", "overhear", "overnight", "overturn", "owe", "owing", "owl", "own", "owner", "ox", "pa", "pace", "pack", "package", "pad", "page", "paid", "pail", "pain", "painful", "paint", "painter", "painting", "pair", "pal", "palace", "pale", "pan", "pancake", "pane", "pansy", "pants", "papa", "paper", "parade", "pardon", "parent", "park", "part", "partly", "partner", "party", "pass", "passenger", "past", "paste", "pasture", "pat", "patch", "path", "patter", "pave", "pavement", "paw", "pay", "payment", "pea", "peace", "peaceful", "peach", "peaches", "peak", "peanut", "pear", "pearl", "peas", "peck", "peek", "peel", "peep", "peg", "pen", "pencil", "penny", "people", "pepper", "peppermint", "perfume", "perhaps", "person", "pet", "phone", "piano", "pick", "pickle", "picnic", "picture", "pie", "piece", "pig", "pigeon", "piggy", "pile", "pill", "pillow", "pin", "pine", "pineapple", "pink", "pint", "pipe", "pistol", "pit", "pitch", "pitcher", "pity", "place", "plain", "plan", "plane", "plant", "plate", "platform", "platter", "play", "player", "playground", "playhouse", "playmate", "plaything", "pleasant", "please", "pleasure", "plenty", "plow", "plug", "plum", "pocket", "pocketbook", "poem", "point", "poison", "poke", "pole", "police", "policeman", "polish", "polite", "pond", "ponies", "pony", "pool", "poor", "pop", "popcorn", "popped", "porch", "pork", "possible", "post", "postage", "postman", "pot", "potato", "potatoes", "pound", "pour", "powder", "power", "powerful", "praise", "pray", "prayer", "prepare", "present", "pretty", "price", "prick", "prince", "princess", "print", "prison", "prize", "promise", "proper", "protect", "proud", "prove", "prune", "public", "puddle", "puff", "pull", "pump", "pumpkin", "punch", "punish", "pup", "pupil", "puppy", "pure", "purple", "purse", "push", "puss", "pussy", "pussycat", "put", "putting", "puzzle", "quack", "quart", "quarter", "queen", "queer", "question", "quick", "quickly", "quiet", "quilt", "quit", "quite", "rabbit", "race", "rack", "radio", "radish", "rag", "rail", "railroad", "railway", "rain", "rainbow", "rainy", "raise", "raisin", "rake", "ram", "ran", "ranch", "rang", "rap", "rapidly", "rat", "rate", "rather", "rattle", "raw", "ray", "reach", "read", "reader", "reading", "ready", "real", "really", "reap", "rear", "reason", "rebuild", "receive", "recess", "record", "red", "redbird", "redbreast", "refuse", "reindeer", "rejoice", "remain", "remember", "remind", "remove", "rent", "repair", "repay", "repeat", "report", "rest", "return", "review", "reward", "rib", "ribbon", "rice", "rich", "rid", "riddle", "ride", "rider", "riding", "right", "rim", "ring", "rip", "ripe", "rise", "rising", "river", "road", "roadside", "roar", "roast", "rob", "robber", "robe", "robin", "rock", "rocket", "rocky", "rode", "roll", "roller", "roof", "room", "rooster", "root", "rope", "rose", "rosebud", "rot", "rotten", "rough", "round", "route", "row", "rowboat", "royal", "rub", "rubbed", "rubber", "rubbish", "rug", "rule", "ruler", "rumble", "run", "rung", "runner", "running", "rush", "rust",
        "rusty", "rye", "sack", "sad", "saddle", "sadness", "safe", "safety", "said", "sail", "sailboat", "sailor", "saint", "salad", "sale", "salt", "same", "sand", "sandwich", "sandy", "sang", "sank", "sap", "sash", "sat", "satin", "satisfactory", "saturday", "sausage", "savage", "save", "savings", "saw", "say", "scab", "scales", "scare", "scarf", "school", "schoolboy", "schoolhouse", "schoolmaster", "schoolroom", "scorch", "score", "scrap", "scrape", "scratch", "scream", "screen", "screw", "scrub", "sea", "seal", "seam", "search", "season", "seat", "second", "secret", "see", "seed", "seeing", "seek", "seem", "seen", "seesaw", "select", "self", "selfish", "sell", "send", "sense", "sent", "sentence", "separate", "september", "servant", "serve", "service", "set", "setting", "settle", "settlement", "seven", "seventeen", "seventh", "seventy", "several", "sew", "shade", "shadow", "shady", "shake", "shaker", "shaking", "shall", "shame", "shan't", "shape", "share", "sharp", "shave", "she", "she'd", "she'll", "she's", "shear", "shears", "shed", "sheep", "sheet", "shelf", "shell", "shepherd", "shine", "shining", "shiny", "ship", "shirt", "shock", "shoe", "shoemaker", "shone", "shook", "shoot", "shop", "shopping", "shore", "short", "shot", "should", "shoulder", "shouldn't", "shout", "shovel", "show", "shower", "shut", "shy", "sick", "sickness", "side", "sidewalk", "sideways", "sigh", "sight", "sign", "silence", "silent", "silk", "sill", "silly", "silver", "simple", "sin", "since", "sing", "singer", "single", "sink", "sip", "sir", "sis", "sissy", "sister", "sit", "sitting", "six", "sixteen", "sixth", "sixty", "size", "skate", "skater", "ski", "skin", "skip", "skirt", "sky", "slam", "slap", "slate", "slave", "sled", "sleep", "sleepy", "sleeve", "sleigh", "slept", "slice", "slid", "slide", "sling", "slip", "slipped", "slipper", "slippery", "slit", "slow", "slowly", "sly", "smack", "small", "smart", "smell", "smile", "smoke", "smooth", "snail", "snake", "snap", "snapping", "sneeze", "snow", "snowball", "snowflake", "snowy", "snuff", "snug", "so", "soak", "soap", "sob", "socks", "sod", "soda", "sofa", "soft", "soil", "sold", "soldier", "sole", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "son", "song", "soon", "sore", "sorrow", "sorry", "sort", "soul", "sound", "soup", "sour", "south", "southern", "space", "spade", "spank", "sparrow", "speak", "speaker", "spear", "speech", "speed", "spell", "spelling", "spend", "spent", "spider", "spike", "spill", "spin", "spinach", "spirit", "spit", "splash", "spoil", "spoke", "spook", "spoon", "sport", "spot", "spread", "spring", "springtime", "sprinkle", "square", "squash", "squeak", "squeeze", "squirrel", "stable", "stack", "stage", "stair", "stall", "stamp", "stand", "star", "stare", "start", "starve", "state", "states", "station", "stay", "steak", "steal", "steam", "steamboat", "steamer", "steel", "steep", "steeple", "steer", "stem", "step", "stepping", "stick", "sticky", "stiff", "still", "stillness", "sting", "stir", "stitch", "stock", "stocking", "stole", "stone", "stood", "stool", "stoop", "stop", "stopped", "stopping", "store", "stories", "stork", "storm", "stormy", "story", "stove", "straight", "strange", "stranger", "strap", "straw", "strawberry", "stream", "street", "stretch", "string", "strip", "stripes", "strong", "stuck", "study", "stuff", "stump", "stung", "subject", "such", "suck", "sudden", "suffer", "sugar", "suit", "sum", "summer", "sun", "sunday", "sunflower", "sung", "sunk", "sunlight", "sunny", "sunrise", "sunset", "sunshine", "supper", "suppose", "sure", "surely", "surface", "surprise", "swallow", "swam", "swamp", "swan", "swat", "swear", "sweat", "sweater", "sweep", "sweet", "sweetheart", "sweetness", "swell", "swept", "swift", "swim", "swimming", "swing", "switch", "sword", "swore", "table", "tablecloth", "tablespoon", "tablet", "tack", "tag", "tail", "tailor", "take", "taken", "taking", "tale", "talk", "talker", "tall", "tame", "tan", "tank", "tap", "tape", "tar", "tardy", "task", "taste", "taught", "tax", "tea", "teach", "teacher", "team", "tear", "tease", "teaspoon", "teeth", "telephone", "tell", "temper", "ten", "tennis", "tent", "term", "terrible", "test", "than", "thank", "thankful", "thanks", "thanksgiving", "that", "that's", "the", "theater", "thee", "their", "them", "then", "there", "these", "they", "they'd", "they'll", "they're", "they've", "thick", "thief", "thimble", "thin", "thing", "think", "third", "thirsty", "thirteen", "thirty", "this", "tho", "thorn", "those", "though", "thought", "thousand", "thread", "three", "threw", "throat", "throne", "through", "throw", "thrown", "thumb", "thunder", "thursday", "thy", "tick", "ticket", "tickle", "tie", "tiger", "tight", "till", "time", "tin", "tinkle", "tiny", "tip", "tiptoe", "tire", "tired", "tis", "title", "to", "toad", "toadstool", "toast", "tobacco", "today", "toe", "together", "toilet", "told", "tomato", "tomorrow", "ton", "tone", "tongue", "tonight", "too", "took", "tool", "toot", "tooth", "toothbrush", "toothpick", "top", "tore", "torn", "toss", "touch", "tow", "toward", "towards", "towel", "tower", "town", "toy", "trace", "track", "trade", "train", "tramp", "trap", "tray", "treasure", "treat", "tree", "trick", "tricycle", "tried", "trim", "trip", "trolley", "trouble", "truck", "true", "truly", "trunk", "trust", "truth", "try", "tub", "tuesday", "tug", "tulip", "tumble", "tune", "tunnel", "turkey", "turn", "turtle", "twelve", "twenty", "twice", "twig", "twin", "two", "ugly", "umbrella", "uncle", "under", "understand", "underwear", "undress", "unfair", "unfinished", "unfold", "unfriendly", "unhappy", "unhurt", "uniform", "united", "unkind", "unknown", "unless", "unpleasant", "until", "unwilling", "up", "upon", "upper", "upset", "upside", "upstairs", "uptown", "upward", "us", "use", "used", "useful", "valentine", "valley", "valuable", "value", "vase", "vegetable", "velvet", "very", "vessel", "victory", "view", "village", "vine", "violet", "visit", "visitor", "voice", "vote", "wag", "wagon", "waist", "wait", "wake", "waken", "walk", "wall", "walnut", "want", "war", "warm", "warn", "was", "wash", "washer", "washtub", "wasn't", "waste", "watch", "watchman", "water", "watermelon", "waterproof", "wave", "wax", "way", "wayside", "we", "we'd", "we'll", "we're", "we've", "weak", "weaken", "weakness", "wealth", "weapon", "wear", "weary", "weather", "weave", "web", "wedding", "wednesday", "wee", "weed", "week", "weep", "weigh", "welcome", "well", "went", "were", "west", "western", "wet", "whale", "what", "what's", "wheat", "wheel", "when", "whenever", "where", "which", "while", "whip", "whipped", "whirl", "whiskey", "whisky", "whisper", "whistle", "white", "who", "who'd", "who'll", "who's", "whole", "whom", "whose", "why", "wicked", "wide", "wife", "wiggle", "wild", "wildcat", "will", "willing", "willow", "win", "wind", "windmill", "window", "windy", "wine", "wing", "wink", "winner", "winter", "wipe", "wire", "wise", "wish", "wit", "witch", "with", "without", "woke", "wolf", "woman", "women", "won", "won't", "wonder", "wonderful", "wood", "wooden", "woodpecker", "woods", "wool", "woolen", "word", "wore", "work", "worker", "workman", "world", "worm", "worn", "worry", "worse", "worst", "worth", "would", "wouldn't", "wound", "wove", "wrap", "wrapped", "wreck", "wren", "wring", "write", "writing", "written", "wrong", "wrote", "wrung", "yard", "yarn", "year", "yell", "yellow", "yes", "yesterday", "yet", "yolk", "yonder", "you", "you'd", "you'll", "you're", "you've", "young", "youngster", "your", "yours", "yourself", "yourselves", "youth"
    ])
    TOKENIZER = RegexpTokenizer(r"\w+'\w+|\w+")
    SPECIAL_CHAR = '@<:@'

    def __init__(self, text):
        self.raw_text = text
        self.raw_text_length = len(text)
        self.number_of_letters = len([x for x in self.raw_text if x.isalpha() or x.isdigit()])
        self.words = StylometryExtractor.TOKENIZER.tokenize(self.raw_text)
        self.tokens = word_tokenize(self.raw_text)
        self.number_of_words = len(self.words)
        self.number_of_tokens = len(self.tokens)
#         self.text = Text(word_tokenize(self.raw_text))
        self.words_frequency = FreqDist(Text(self.words))
        self.tokens_frequency = FreqDist(Text(self.tokens))
        self.sentences = sent_tokenize(self.raw_text)
        self.number_of_sentences = len(self.sentences)
        self.sentence_chars = [len(sent) for sent in self.sentences]
        self.sentence_word_length = [len(sent.split()) for sent in self.sentences]
        self.paragraphs = [p for p in self.raw_text.split("\n\n") if len(p) > 0 and not p.isspace()]
        self.paragraph_word_length = [len(p.split()) for p in self.paragraphs]
        self.all_trigrams = self._all_trigrams()
        self.ngram_string = self._to_ngram_string()

    def _to_ngram_string(self):
        return StylometryExtractor.SPECIAL_CHAR.join(
            ''.join(ngram) for ngram in ngrams(self.raw_text.lower(), 4) if ' ' not in ngram and '\n' not in ngram)

    def term_per_thousand(self, term):
        return self.words_frequency[term] * 1000 / self.words_frequency.N()

    def char_per_thousand(self, char):
        return self.raw_text.count(char) / self.raw_text_length * 1000

    def chars_per_thousand(self, chars):
        return sum([self.char_per_thousand(char) for char in chars])

    def syllables_per_thousand(self):
        return self.get_number_syllables() / self.raw_text_length * 1000

    def get_number_syllables(self):
        dic = pyphen.Pyphen(lang='en')
        return sum([len(dic.inserted(word).split("-")) for word in self.words])

    def get_number_pollisyllable_words(self):
        dic = pyphen.Pyphen(lang='en')
        return len([word for word in self.words if len(dic.inserted(word).split("-")) >= 3])

    def get_words_longer_than_X(self, x):
        return len([word for word in self.words if len(word) >= x])

    def mean_of_syllables_per_word(self):
        return self.get_number_syllables() / self.number_of_words

    def num_of_words_with_more_than_three_syllables_per_thousand(self):
        return self.get_number_pollisyllable_words() / self.number_of_words * 1000

    def get_flesch_reading_ease(self):
        # http://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
        """
        90.0- 100.0 - sily understood by an average 11-year-old student
        60.0 - 70.0 - easily understood by 13- to 15-year-old students
        0.00 - 30.0 -  best understood by university graduates
        """
        return 206.835 - 1.015 * self.number_of_words / self.number_of_sentences - 84.6 * self.get_number_syllables() / self.number_of_words

    def flesch_kincaid_grade_level(self):
        # http://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
        """
            It is more or less the number of years of education generally required to understand this text.
            The lowest grade level score in theory is -3.40.
        """
        return 0.39 * self.number_of_words / self.number_of_sentences + 11.8 * self.get_number_syllables() / self.number_of_words - 15.59

    def get_coleman_liau_index(self):
        # http://en.wikipedia.org/wiki/Coleman%E2%80%93Liau_index
        """
             It approximates the U.S. grade level thought necessary to comprehend the text.
        """
        return 5.89 * self.number_of_letters / self.number_of_words - 29.6 * self.number_of_sentences / self.number_of_words - 15.8

    def get_gunning_fog_index(self):
        # http://en.wikipedia.org/wiki/Gunning_fog_index
        """
        The index estimates the years of formal education needed to understand the text on a first reading
        """
        return 0.4 * (self.number_of_words / self.number_of_sentences + 100.0 * self.get_number_pollisyllable_words() / self.number_of_words)

    def get_smog_index(self):
        # http://en.wikipedia.org/wiki/SMOG
        """
            Simple Measure of Gobbledygook (SMOG) is a simplification of Gunning Fog, also estimating the years of formal education needed
            to understand a text
        """
        return 1.043 * math.sqrt(self.get_number_pollisyllable_words() * 30.0 / self.number_of_sentences) + 3.1291

    def get_ari_index(self):
        # http://en.wikipedia.org/wiki/Automated_Readability_Index
        """
            It produces an approximate representation of the US grade level needed to comprehend the text.
        """
        return 4.71 * self.number_of_letters / self.number_of_words + 0.5 * self.number_of_words / self.number_of_sentences - 21.43

    def get_lix_index(self):
        # http://en.wikipedia.org/wiki/LIX
        # http://www.readabilityformulas.com/the-LIX-readability-formula.php
        """
            Value interpretation:
            Very Easy      - 20, 25
            Easy           - 30, 35
            Medium         - 40. 45
            Difficult      - 50, 55
            Very Difficult - 60+
        """
        long_words = self.get_words_longer_than_X(6)
        number_of_periods = self.number_of_sentences + self.tokens_frequency[':'] + self.tokens_frequency[';']
        return self.number_of_words / number_of_periods + 100.0 * long_words / self.number_of_words

    def number_of_dale_chall_difficult_words(self):
        return len([word for word in self.words if word not in StylometryExtractor.DALE_CHALL_WORDS])

    def get_dale_chall_score(self):
        # http://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula
        """
            4.9 or lower    ---  easily understood by an average 4th-grade student or lower
            5.0–5.9         ---  easily understood by an average 5th or 6th-grade student
            6.0–6.9         ---  easily understood by an average 7th or 8th-grade student
            7.0–7.9         ---  easily understood by an average 9th or 10th-grade student
            8.0–8.9         ---  easily understood by an average 11th or 12th-grade student
            9.0–9.9         ---  easily understood by an average 13th to 15th-grade (college) student
            10.0 or higher  ---  easily understood by an average college graduate
        """
        return 15.79 * self.number_of_dale_chall_difficult_words() / self.number_of_words + 0.0496 * self.number_of_words / self.number_of_sentences

    def get_dale_chall_known_fraction(self):
        """
            Computes the fraction of easy words in the text, i.e., the fraction of words that could be found in the
            dale chall list of 3.000 easy words.
        """
        return 1.0 - self.number_of_dale_chall_difficult_words() / self.number_of_words

    def mean_sentence_len(self):
        return np.mean(self.sentence_word_length)

    def std_sentence_len(self):
        return np.std(self.sentence_word_length)

    def mean_paragraph_len(self):
        return np.mean(self.paragraph_word_length)

    def std_paragraph_len(self):
        return np.std(self.paragraph_word_length)

    def mean_word_len(self):
        word_chars = [len(word) for word in self.words]
        return sum(word_chars) / len(word_chars)

    def unique_words_ratio(self):
        return len(set(self.words)) / self.number_of_words * 100

#     def get_byte_ngrams(self, number_of_bytes):
    @classmethod
    def to_pos_tags(cls, sentence):
        tokens = StylometryExtractor.TOKENIZER.tokenize(sentence)
        pos_tags = list(map(lambda x: x[1], pos_tag(tokens)))
        return ['__START__'] + pos_tags + ['__END__']

    @classmethod
    def pos_tag_trigrams(cls, sentence):
        pos_tags = StylometryExtractor.to_pos_tags(sentence)
        return [(x, y, z) for x, y, z in zip(pos_tags, pos_tags[1:], pos_tags[2:])]

    def _all_trigrams(self):
        return Counter(trigram
            for sentence in self.sentences
            for trigram in StylometryExtractor.pos_tag_trigrams(sentence)
        )

    def pos_tag_percents(self):
        number_of_trigrams = sum(self.all_trigrams.values())
        return {
            '_'.join(trigram): self.all_trigrams[trigram] / number_of_trigrams
            for trigram in MOST_COMMON_POS_TAG_TRIGRAMS
        }

    def to_dict(self):
        features = {
            'Lexical diversity' : self.unique_words_ratio(),
            'Mean Word Length' : self.mean_word_len(),
            'Mean Sentence Length' : self.mean_sentence_len(),
            'STDEV Sentence Length' : self.std_sentence_len(),
            'Mean paragraph Length' : self.mean_paragraph_len(),
            'Number of letters' : self.number_of_letters,
            'Flesch Reading Ease' : self.get_flesch_reading_ease(),
            'Flesch Kincaid Grade' : self.flesch_kincaid_grade_level(),
            'Coleman Liau Index' : self.get_coleman_liau_index(),
            'Gunning Fog Index' : self.get_gunning_fog_index(),
            'Smog Index' : self.get_smog_index(),
            'Ari Index' : self.get_ari_index(),
            'Lix Index' : self.get_lix_index(),
            'Dale Chall Score' : self.get_dale_chall_score(),
            'Dale Chall Known Fraction' : self.get_dale_chall_known_fraction(),
            'Punctuation' : self.chars_per_thousand(['.', ',', '!', ';', '?']),
            'Special characters' : self.chars_per_thousand(['%', '#', ')', '(', '@', '$', '^','&', '>', '<', '*', '_', '-','=', '-', '+', '/','\\', '\'', '"', '`']),
            'Commas' : self.term_per_thousand(','),
            'Semicolons' : self.term_per_thousand(';'),
            'Quotations' : self.term_per_thousand('\"'),
            'Exclamations' : self.term_per_thousand('!'),
            'Colons' : self.term_per_thousand(':'),
            'Hyphens' : self.term_per_thousand('-'),
            'Double Hyphens' : self.term_per_thousand('--'),
            'A' : self.char_per_thousand('a'),
            'B' : self.char_per_thousand('b'),
            'C' : self.char_per_thousand('c'),
            'D' : self.char_per_thousand('d'),
            'E' : self.char_per_thousand('e'),
            'F' : self.char_per_thousand('f'),
            'G' : self.char_per_thousand('g'),
            'H' : self.char_per_thousand('h'),
            'I' : self.char_per_thousand('i'),
            'J' : self.char_per_thousand('j'),
            'K' : self.char_per_thousand('k'),
            'L' : self.char_per_thousand('l'),
            'M' : self.char_per_thousand('m'),
            'N' : self.char_per_thousand('n'),
            'O' : self.char_per_thousand('o'),
            'P' : self.char_per_thousand('p'),
            'Q' : self.char_per_thousand('q'),
            'R' : self.char_per_thousand('r'),
            'S' : self.char_per_thousand('s'),
            'T' : self.char_per_thousand('t'),
            'U' : self.char_per_thousand('u'),
            'V' : self.char_per_thousand('v'),
            'W' : self.char_per_thousand('w'),
            'X' : self.char_per_thousand('x'),
            'Y' : self.char_per_thousand('y'),
            'Z' : self.char_per_thousand('z'),
            'Numbers' : self.chars_per_thousand(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']),
            'Syllables' : self.syllables_per_thousand(),
            'Mean syllables per word' : self.mean_of_syllables_per_word(),
            'Words with >= 3 syllables' : self.num_of_words_with_more_than_three_syllables_per_thousand(),
            }

        for stopword in stopwords.words('english'):
            features[stopword] = self.term_per_thousand(stopword)

        features.update(self.pos_tag_percents())

        return features
