import pygame
from random import randint, choices, choice
import mido
from music21 import converter, chord, stream, midi, note
from pychord import Chord



CHORD_NAMES = []
BARS = 1.0
CHORDS_PER_MELODY = BARS * 4
NOTE_PER_CHORD = 3
INPUT_NPQ = [] #< notes in each quarterlength
INPUT_OPQ=[] #< minimum octave in each quarterlength

BPM = 125
VOLUME = 100
TRACK = 0
CHANNEL = 0
ALPHA = 500
INPUT_KEY = ""
INPUT_KEY_CC = 0
INPUT_FIRST_NOTE = ""
INPUT_LAST_NOTE = ""
INPUT_OCTAVE = ""
INPUT_STREAM = stream.Stream()
REST_TYPE = type(note.Rest())
CHORD_PROGRESSION = {}



def generate_chromosome():
    return choice(CHORD_NAMES)


def generate_individual():
    return [generate_chromosome() for _ in range(CHORDS_PER_MELODY)]


def generate_population(size: int):
    return [generate_individual() for _ in range(size)]


def similarity_score(indv):
    score = 0
    try:
        s = individual_to_stream(indv)
        if str(s.analyze('key')) == INPUT_KEY:
            score = ALPHA - ALPHA * abs(INPUT_KEY_CC - s.analyze('key').correlationCoefficient)
            return score
        for generated_key in s.analyze('key').alternateInterpretations:
            if str(generated_key) == INPUT_KEY:
                score = ALPHA - ALPHA * abs(INPUT_KEY_CC - generated_key.correlationCoefficient)
                return score
    except Exception as ex:
        print(ex)

    return score
def get_freqeuencies(note_list):
    freq = []
    for n in note_list:
        if n == 'rest':
            freq.append(0)
        else:
            freq.append(note.Note(n).pitch.freq440)
    return freq
def hm_note(chord_name, beat_number):
    score = 0
    input_freq = get_freqeuencies(INPUT_NPQ[beat_number])
    gen_freq = get_freqeuencies(Chord(chord_name).components())

    for f in gen_freq:
        if f in input_freq:
            score += 1

    return score
def note_intersection_score(indv):
    score = 0
    cnt = 0
    for i in range(CHORDS_PER_MELODY):
        cnt += hm_note(indv[i], i)
        #print(f"{i}, {CHORDS_PER_MELODY}")

    score = 2 * ALPHA * cnt/(CHORDS_PER_MELODY * 3)
    return score

def remove_rep_sus(indv):
    prev = indv[0]
    lst = []
    if 'sus' not in prev:
        lst.append(prev)
    for i in range(len(indv) - 1):
        if indv[i + 1] == prev:
            continue
        if 'sus' not in indv[i + 1]:
            lst.append(indv[i + 1])
        prev = indv[i + 1]
    return lst


def correct_chord_progression_score(indv):
    score = 0
    progression = CHORD_PROGRESSION[INPUT_KEY]
    tonic = [progression[0] , progression[2]]
    dom = [progression[4], progression[6]]
    subDom = [progression[1], progression[3], progression[5]]
    broke = 0
    prev = 0
    gen = remove_rep_sus(indv)
    prev = -1
    idx = 0
    for crd in gen:
        if crd in tonic:
            prev = 0
            idx += 1
            break
        if crd in dom:
            prev = 1
            broke += 1
            idx += 1
            break
        if crd in subDom:
            prev = 2
            broke += 1
            idx += 1
            break
        else:
            broke += 1
            idx += 1
    if prev == -1:
        return ALPHA * (1 - (broke/len(gen)))

    while idx < len(gen):
        cur = 0
        if gen[idx] in tonic:
            cur = 0
        if gen[idx] in dom:
            cur = 1
        if gen[idx] in subDom:
            cur = 2
        else:
            broke += 1
            idx += 1
            continue
        if prev == 1 and cur != 0:
            broke += 1
        if prev == 2 and cur != 1:
            broke += 1
        prev = cur
        idx += 1

    score = ALPHA * (1 - (broke/len(gen)))
    return score


def fitness(indv):
    sm_score = similarity_score(indv)
    ni_score = note_intersection_score(indv)
    ccp_score = correct_chord_progression_score(indv)

    return sm_score + ni_score + ccp_score

def avoid_negatives(pop):
    constant = 5
    w = [fitness(indv) for indv in pop]
    smallest_weight = abs(min(w))
    for i in range(len(w)):
        w[i] += smallest_weight + constant
    return w

def selection_pair(pop):

    w = avoid_negatives(pop)
    return choices(
        population=pop,
        weights=w,
        k=2
    )


def single_point_crossover(a, b):
    length = len(a)
    p = randint(1, length - 1)
    return a[0:p] + b[p:], b[0: p] + a[p:]


def mutation(indv, probability: float = 0.5):
    index = 0
    for _ in indv:
        if randint(1, 11) >= probability * 10:
            indv[index] = generate_chromosome()
        index += 1
    return indv

def population_fitness(pop):
    total = 0
    best = -1e6
    worst = 1e6
    for indv in pop:
        k = fitness(indv)
        total += k
        if k < worst:
            worst = k
        if k > best:
            best = k
    return total,best, worst

def print_progress(pop, gen_num):
    total,best,worst = population_fitness(pop)
    print(f"Generation number {gen_num}")
    print("==============")
    print("Average fitness of population is %f"%(total/ len(pop)))
    print(f"Best is {best}, worst is{worst}")


def run_evolution(
        fitness_limit: int,
        generation_limit: int,
        population_size: int,
) :

    pop = generate_population(population_size)

    for i in range(generation_limit):
        pop = sorted(
            pop,
            key=fitness,
            reverse=True
        )
        print_progress(pop, i)
        if fitness(pop[0]) >= fitness_limit:
            break
        #
        next_generation = pop[0: 2]
        print("Constructing new generation!")
        for j in range(int(len(pop) / 2) - 1):
            parents = selection_pair(pop)
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            offspring_a = mutation(offspring_a)
            offspring_b = mutation(offspring_b)
            next_generation += [offspring_a, offspring_b]
        print("Done with new generation!")
        pop = next_generation

        print("sorting next generation")
        pop = sorted(
            pop,
            key=fitness,
            reverse=True
        )
        print("sorted")

    return pop, i


def play_music(midi_file):
    mid = mido.MidiFile(midi_file, clip=True)
    pygame.mixer.init(44100, -16, 2, 1024)
    clock = pygame.time.Clock()
    pygame.mixer.music.load(midi_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        clock.tick(30)





def individual_to_stream(indv):
    s = stream.Stream()
    lst = []
    for i in range(len(indv)):
        for n in Chord(indv[i]).components():
            lst.append((str(n) + str(INPUT_OPQ[i] - 2)))
        s.append(chord.Chord(lst))
        lst.clear()
    return s


def individual_to_midi(indv, input_number):
    generated_stream = individual_to_stream(indv)
    crd = generated_stream.chordify()
    input = converter.parse(f'data/input{str(input_number)}.mid')
    input.append(generated_stream)
    input.write('midi', f'output/output{str(input_number)}.mid')
    crd.write('midi', f'generated/gen{str(input_number)}.mid')


def analyze_input(input_number):
    global INPUT_STREAM
    INPUT_STREAM = converter.parse(f'data/input{str(input_number)}.mid')
    global CHORDS_PER_MELODY
    global INPUT_KEY
    INPUT_KEY = str(INPUT_STREAM.analyze('key'))

    global INPUT_KEY_CC
    INPUT_KEY_CC = INPUT_STREAM.analyze('key').correlationCoefficient
    global INPUT_NPQ
    global INPUT_OCTAVE
    mini = 10
    i = 0.0
    lst = list()
    for n in INPUT_STREAM.flat.notesAndRests:
        i += float(n.duration.quarterLength)
        if type(n) == REST_TYPE:
            lst.append('rest')
            if i == 1:
                copy = lst.copy()
                INPUT_NPQ.append(copy)
                i = 0
                lst.clear()
                INPUT_OPQ.append(mini)
                mini = 10
                continue
            if i > 1:
                copy = lst.copy()
                INPUT_NPQ.append(copy)
                i -= 1
                lst.clear()
                lst.append('rest')
                INPUT_OPQ.append(mini)
                mini = 10
        if type(n) != REST_TYPE:
            lst.append(str(n.name))
            if mini > n.octave:
                mini = n.octave

            if i == 1:
                copy = lst.copy()
                INPUT_NPQ.append(copy)
                i = 0
                lst.clear()
                INPUT_OPQ.append(mini)
                mini = 10
                continue
            if i > 1:
                copy = lst.copy()
                INPUT_NPQ.append(copy)
                i -= 1
                lst.clear()
                lst.append(str(n.name))
                m = mini
                INPUT_OPQ.append(m)
                mini = 10
    CHORDS_PER_MELODY = len(INPUT_OPQ)




    return


def initialize_constants():
    global CHORD_NAMES
    global CHORD_PROGRESSION
    CHORD_NAMES += ["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb",
                    "G", "G#", "Ab", "A", "A#", "Bb", "B"]  # <- major chords
    size = len(CHORD_NAMES)
    # adding minor chords
    for i in range(size):
        CHORD_NAMES.append(CHORD_NAMES[i] + "m")

    # adding sus2

    CHORD_NAMES += ["Csus2", "C#sus2", "Dsus2", "Ebsus2", "Esus2", "Fsus2", "F#sus2",
                    "Gsus2", "Absus2", "Asus2", "Bbsus2", "Bsus2"]

    # adding sus4
    CHORD_NAMES += ["Csus4", "C#sus4", "Dsus4", "Ebsus4", "Esus4", "Fsus4", "F#sus4",
                    "Gsus4", "Absus4", "Asus4", "Bbsus4", "Bsus4"]
    # adding dim chords
    CHORD_NAMES += ['Cdim', 'C#dim', 'Dbdim', 'Ddim', 'D#dim', 'Ebdim', 'Edim', 'Fdim', 'F#dim', 'Gbdim', 'Gdim','G#dim','Adim' ,'Abdim', 'A#dim', 'Bbdim', 'Bdim']

    CHORD_PROGRESSION = {
        "C major" : ["C", "Dm", "Em", "F", "G", "Am", "Bdim"],
        "D major" : ["D", "Em", "F#m", "G", "A", "Bm", "C#dim"],
        "E- major" :["Eb", "Fm", "Gm", "Ab", "Bb", "Cm", "Ddim"],
        "E major" : ["E", "F#m", "G#m", "A", "B", "C#m", "D#dim"],
        "F major" : ["F", "Gm", "Am", "Bb", "C", "Dm", "Edim"],
        "G major" : ["G", "Am", "Bm", "C", "D", "Em", "F#dim"],
        "A- major" : ["Ab", "Bbm", "Cm", "Db", "Eb", "Fm", "Gdim"],
        "A major" : ["A", "Bm", "C#m", "D", "E", "F#m", "G#dim"],
        "B- major" : ["Bb", "Cm", "Dm", "Eb", "F", "Dm", "Adim"],
        "B major" : ["B", "C#m", 'D#m', "E", "F#", "G#m", "A#dim"],

        "c minor": ["Cm", "Ddim", "Eb", "Fm", "Gm", "Ab", "Bb"],
        "c# minor" : ["C#m", "D#dim", "E", "F#m", "G#m", "A", "B"],
        "d minor": ["Dm", "Edim", "F", "Gm", "Am", "Bb", "C"],
        "e minor": ["Em", "F#dim", "G", "Am", "Bm", "C", "D"],
        "f minor": ["Fm", "Gdim", "Ab", "Bbm", "Cm", "Db", "Eb"],
        "f# minor" : ["F#m", "G#dim", "A", "Bm", "C#m", "D", "E"],
        "g minor": ["Gm", "Adim", "Bb", "Cm", "Dm", "Eb", "F"],
        "a minor": ["Am", "Bdim", "C", "Dm", "Em", "F", "G"],
        "a# minor": ["A#m", "B#dim", "C#", "D#m", "E#m", "F#", "G#"],
        "b minor": ["Bm", "C#dim", 'D', "Em", "F#m", "G", "A"],
    }

    return

def main():
    initialize_constants()
    for x in range(3):
        print(f"Working on input :{x + 1}")
        print("==============")
        analyze_input(x+1)
        pop, gen_num = run_evolution(10000, 10 , 50)
        individual_to_midi(pop[0], x+1)
        play_music(f'output/output{x + 1}.mid')
if __name__ == '__main__':
    main()
