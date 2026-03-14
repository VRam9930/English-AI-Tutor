"""
lessons.py

Lesson content repository for Linguist-OS.
Contains 7 comprehensive English language lessons covering the core
grammar and vocabulary concepts tracked by the mastery system.

Each lesson includes:
  - RULE: The grammar/vocab rule explained simply
  - SIGNAL WORDS: Trigger words that indicate this concept
  - CORRECT EXAMPLES: 3 examples with checkmarks
  - WRONG EXAMPLES: 3 examples with corrections
  - FORMULA: A simple formula to remember the rule

These lessons are stored in ChromaDB at startup for RAG retrieval,
and are also available as a Python list for direct access.
"""

LESSONS = [
    {
        "id": "past_simple",
        "concept": "past_simple",
        "content": """PAST SIMPLE vs PRESENT PERFECT

RULE: Use Past Simple for completed actions at a SPECIFIC time in the past. Use Present Perfect for actions connected to the present or when the time is NOT specified.

SIGNAL WORDS:
Past Simple: yesterday, last week, in 2020, ago, when I was young
Present Perfect: already, yet, just, ever, never, since, for, recently

CORRECT EXAMPLES:
I went to Paris last summer. (specific time = Past Simple)
I have visited Paris three times. (no specific time = Present Perfect)
She finished her homework an hour ago. (specific time = Past Simple)

WRONG EXAMPLES:
I have went to Paris last summer. -> I went to Paris last summer. (specific time needs Past Simple)
I visited Paris three times already. -> I have visited Paris three times already. (connected to present needs Present Perfect)
She has finished her homework an hour ago. -> She finished her homework an hour ago. ('ago' needs Past Simple)

FORMULA: Specific time in past = Past Simple (V2) | No specific time / connected to now = Present Perfect (have/has + V3)""",
    },
    {
        "id": "present_perfect",
        "concept": "present_perfect",
        "content": """PRESENT PERFECT USAGE

RULE: Present Perfect (have/has + past participle) is used for: 1) Experiences in life, 2) Actions that started in the past and continue now, 3) Recent actions with present results, 4) Actions where the exact time is not important.

SIGNAL WORDS:
already, yet, just, ever, never, since, for, recently, so far, up to now, still

CORRECT EXAMPLES:
I have already eaten lunch. (recent action, result = not hungry now)
She has lived here since 2015. (started in past, continues now)
Have you ever been to Japan? (life experience, time not specified)

WRONG EXAMPLES:
I have already ate lunch. -> I have already eaten lunch. (must use past participle 'eaten' not past simple 'ate')
She has live here since 2015. -> She has lived here since 2015. (need past participle after has)
Did you ever been to Japan? -> Have you ever been to Japan? (life experience needs Present Perfect)

FORMULA: have/has + PAST PARTICIPLE (V3) | Use for: experience, duration, recent result, unspecified time""",
    },
    {
        "id": "articles",
        "concept": "articles",
        "content": """ARTICLES: A, AN, THE

RULE: Use 'a/an' for non-specific singular countable nouns (first mention). Use 'the' for specific nouns (known to both speaker and listener). Use no article for general plural nouns and uncountable nouns.

SIGNAL CLUES:
'a/an': first time mentioning, one of many, not specific
'the': both know which one, already mentioned, only one exists, superlatives
No article: general statements, abstract concepts, meals, sports

CORRECT EXAMPLES:
I saw a dog in the park. The dog was very friendly. ('a' first mention, 'the' second mention)
The sun rises in the east. ('the' = only one sun, one east)
She is an engineer. ('an' before vowel sound, non-specific)

WRONG EXAMPLES:
I saw dog in park. -> I saw a dog in the park. (singular countable nouns need articles)
I want to buy the car someday. -> I want to buy a car someday. (non-specific car = 'a')
She is a engineer. -> She is an engineer. (use 'an' before vowel sound)

FORMULA: a/an = non-specific, first mention | the = specific, known, unique | zero = general plural/uncountable""",
    },
    {
        "id": "prepositions",
        "concept": "prepositions",
        "content": """PREPOSITIONS: IN, ON, AT

RULE FOR TIME:
AT = specific times (at 5 PM, at noon, at night, at the weekend)
ON = days and dates (on Monday, on July 4th, on my birthday)
IN = longer periods (in January, in 2020, in the morning, in summer)

RULE FOR PLACE:
AT = specific points (at the bus stop, at home, at school)
ON = surfaces (on the table, on the wall, on the floor)
IN = enclosed spaces (in the room, in the box, in London)

CORRECT EXAMPLES:
I wake up at 7 AM. (specific time)
The meeting is on Wednesday. (day)
She was born in 1995. (year)

WRONG EXAMPLES:
I wake up in 7 AM. -> I wake up at 7 AM. (specific time needs 'at')
The meeting is at Wednesday. -> The meeting is on Wednesday. (day needs 'on')
She was born on 1995. -> She was born in 1995. (year needs 'in')

FORMULA: AT = point (time/place) | ON = surface/day | IN = enclosed/period""",
    },
    {
        "id": "subject_verb",
        "concept": "subject_verb_agreement",
        "content": """SUBJECT-VERB AGREEMENT

RULE: The verb must agree with its subject in number. Singular subjects take singular verbs. Plural subjects take plural verbs. Be careful with: 'everyone/nobody' (singular), 'there is/are', and compound subjects.

KEY RULES:
He/She/It + verb-s (She walks, He runs, It works)
I/You/We/They + base verb (I walk, They run, We work)
Everyone/Nobody/Each = singular (Everyone is here)
There is + singular / There are + plural

CORRECT EXAMPLES:
She goes to school every day. (singular subject 'she' + 'goes')
The students are studying hard. (plural subject 'students' + 'are')
Everyone has finished the test. (everyone = singular + 'has')

WRONG EXAMPLES:
She go to school every day. -> She goes to school every day. (singular 'she' needs 'goes')
The students is studying hard. -> The students are studying hard. (plural 'students' needs 'are')
Everyone have finished the test. -> Everyone has finished the test. ('everyone' is singular, needs 'has')

FORMULA: Singular subject + singular verb (add -s/-es) | Plural subject + base verb""",
    },
    {
        "id": "irregular_verbs",
        "concept": "irregular_verbs",
        "content": """COMMON IRREGULAR VERB FORMS

RULE: Irregular verbs do not follow the regular -ed pattern for past simple and past participle. You must memorize them. The three forms are: Base (V1), Past Simple (V2), Past Participle (V3).

COMMON IRREGULAR VERBS:
go -> went -> gone
do -> did -> done
see -> saw -> seen
take -> took -> taken
write -> wrote -> written
break -> broke -> broken
speak -> spoke -> spoken
eat -> ate -> eaten
give -> gave -> given
come -> came -> come
buy -> bought -> bought
think -> thought -> thought

CORRECT EXAMPLES:
I went to the store yesterday. (past simple of 'go' = 'went')
She has written three books. (past participle of 'write' = 'written')
They have eaten lunch already. (past participle of 'eat' = 'eaten')

WRONG EXAMPLES:
I goed to the store yesterday. -> I went to the store yesterday. ('go' is irregular: went, not goed)
She has wrote three books. -> She has written three books. (past participle is 'written' not 'wrote')
They have ate lunch already. -> They have eaten lunch already. (past participle is 'eaten' not 'ate')

FORMULA: Must memorize V1/V2/V3 forms | Past Simple = V2 | have/has + V3 = Present Perfect""",
    },
    {
        "id": "vocabulary",
        "concept": "vocabulary",
        "content": """VOCABULARY: BASIC TO ADVANCED WORD UPGRADES

RULE: Expanding your vocabulary makes your English more precise, natural, and impressive. Replace common basic words with more specific, advanced alternatives that match the context.

UPGRADE EXAMPLES:
good -> excellent, outstanding, remarkable, superb
bad -> terrible, dreadful, appalling, inadequate
big -> enormous, massive, substantial, significant
small -> tiny, minute, compact, modest
happy -> delighted, thrilled, ecstatic, content
sad -> devastated, heartbroken, melancholy, gloomy
nice -> pleasant, delightful, charming, wonderful
said -> exclaimed, whispered, mentioned, declared
walk -> stroll, march, wander, stride
think -> consider, believe, contemplate, reckon

CORRECT EXAMPLES:
The concert was outstanding! (instead of 'very good')
She was thrilled to receive the award. (instead of 'very happy')
He strolled through the park on a pleasant afternoon. (instead of 'walked' and 'nice')

WRONG EXAMPLES:
The food was very very good. -> The food was exceptional/exquisite. (avoid 'very very', use a stronger word)
I am very tired. -> I am exhausted/drained. (use a single precise word instead of 'very + basic')
It was a big problem. -> It was a significant/major problem. (more precise than 'big')

FORMULA: Avoid 'very + basic adjective' | Use one precise advanced word instead | Match formality to context""",
    },
]
