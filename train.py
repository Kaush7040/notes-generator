import nltk
nltk.download('punkt_tab')
from transformers import pipeline

# Ensure nltk sentence tokenizer is downloaded
nltk.download('punkt')

# Load the model (Using Flan-T5-XL for better summaries)
summarizer = pipeline("summarization", model="google/flan-t5-xl")

def split_text_by_sentences(text, max_words=300):
    """Splits text into chunks while maintaining sentence integrity."""
    sentences = nltk.sent_tokenize(text)
    chunks, chunk = [], []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) > max_words:
            chunks.append(" ".join(chunk))
            chunk = []
            word_count = 0
        chunk.append(sentence)
        word_count += len(words)

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks

def summarize_text(text):
    """Summarizes large text into structured bullet points."""
    chunks = split_text_by_sentences(text)
    summaries = []

    for chunk in chunks:
        prompt = f"""
        Summarize the following text into a structured, detailed set of bullet points. 
        Each bullet should cover a key concept and include supporting details or examples.

        Text: {chunk}
        """
        summary = summarizer(prompt, max_length=300, min_length=200, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    return format_bullet_points(" ".join(summaries))

def format_bullet_points(text):
    """Formats output text into structured bullet points."""
    bullets = text.split(". ")
    formatted = []
    
    for point in bullets:
        if ":" in point:  # Detects main topics
            formatted.append(f"\n🔹 **{point.strip()}**")
        else:
            formatted.append(f"   - {point.strip()}")  # Sub-point

    return "\n".join(formatted)

if __name__ == "__main__":
    # Example large text input
    text = """The Viking village of Berk is frequently attacked by dragons that steal livestock and endanger the villagers. Hiccup, the 15-year-old son of the village chieftain, Stoick the Vast, is deemed too weak to fight. Instead, he creates mechanical devices under apprenticeship with Gobber, the village blacksmith. Hiccup uses a bolas launcher to shoot down a Night Fury, a rare dragon, during a dragon raid, but nobody believes him. He enters the forest and finds the creature but cannot bring himself to kill it. Instead, he sets the dragon free. The Night Fury then suddenly pins Hiccup down, and he braces for death. However, much to Hiccup's surprise, it spares him.Before leaving with his fleet to find and destroy the dragons' nest, Stoick enrolls Hiccup in a dragon-fighting class with fellow teenagers Fishlegs, Snotlout, twins Ruffnut and Tuffnut, and Astrid, on whom Hiccup has a crush. Facing little success in the class, Hiccup returns to the forest and finds the Night Fury in a cove, unable to fly because Hiccup's bolas tore off half of its t
ail fin. Hiccup gradually befriends the dragon, naming him "Toothless" after his retractable teeth, and designs a harness and prosthetic fin that allows Toothless to fly with Hiccup riding him.Learning dragon behavior from Toothless, Hiccup can subdue the captive dragons during training, earning admiration from his peers but sparking suspicion and jealousy from Astrid. Stoick's fleet returns home unsuccessful after being destroyed by a massive dragon. Hiccup must kill a dragon for his final exam. He tries to run away with Toothless, but Astrid discovers the dragon. Hiccup takes her on a flight to demonstrate that Toothless is friendly. During the flight, Toothless is hypnotically drawn to the dragons' nest. There, a gargantuan dragon named the Red Death summons smaller dragons to feed it copious amounts of live food to avoid being eaten themselves. Realizing the dragons have been forced to attack Berk to survive, Astrid wishes to tell the village, but Hiccup advises against it to protect Toothless.In his final exam, Hiccup faces a captive Monstrous Nightmare dragon and tries to subdue him to prove that dragons can be peaceful. When Stoick unintentionally enrages the dragon into attacking, Toothless arrives to protect Hiccup but is captured. Stoick furiously confronts his son for befriending a dragon until Hiccup accidentally reveals that Toothless knows the location of the dragons' nest. Stoick reluctantly disowns Hiccup and sets off for the nest with Toothless guiding the Vikings, in spite of Hiccup's pleas. Astrid prompts Hiccup to realize he spared Toothless out of compassion, not weakness. Regaining his confidence, Hiccup shows his friends how to befriend the training dragons, and they set out after Toothless.Stoick and his Vikings locate and break open the dragon's nest, awakening the Red Death, which easily overwhelms them. Hiccup and his friends ride in on the training dragons, distracting the Red Death. Hiccup attempts to free Toothless; Stoick rescues and apologizes to both Hiccup and Toothless, and reinstates Hiccup as his son. Toothless and Hiccup destroy the Red Death through teamwork. Whilst escaping
 the explosion, Hiccup gets knocked off Toothless. The Vikings and Stoick discover that Toothless saved Hiccup from the explosion by covering him with his wings, but Hiccup has lost his lower left leg.Sometime later, Hiccup awakes back on Berk and finds that Gobber has fashioned him a prosthetic. He is now admired by his village, including Astrid, who kisses him. Berk begins a new era of humans and dragons living in peace.
    
    """

    summary = summarize_text(text)
    print(summary)