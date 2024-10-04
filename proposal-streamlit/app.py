import streamlit as st
import streamlit.components.v1 as components
import numpy as np

# Custom CSS for futuristic design
def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("./style.css")

# Particle animation background
def particle_background():
    components.html(
        """
        <div id="particles-js"></div>
        <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
        <script>
            particlesJS("particles-js", {
                "particles": {
                    "number": {"value": 80, "density": {"enable": true, "value_area": 800}},
                    "color": {"value": "#ffffff"},
                    "shape": {"type": "circle", "stroke": {"width": 0, "color": "#000000"}, "polygon": {"nb_sides": 5}},
                    "opacity": {"value": 0.5, "random": false, "anim": {"enable": false, "speed": 1, "opacity_min": 0.1, "sync": false}},
                    "size": {"value": 3, "random": true, "anim": {"enable": false, "speed": 40, "size_min": 0.1, "sync": false}},
                    "line_linked": {"enable": true, "distance": 150, "color": "#ffffff", "opacity": 0.4, "width": 1},
                    "move": {"enable": true, "speed": 6, "direction": "none", "random": false, "straight": false, "out_mode": "out", "bounce": false, "attract": {"enable": false, "rotateX": 600, "rotateY": 1200}}
                },
                "interactivity": {
                    "detect_on": "canvas",
                    "events": {"onhover": {"enable": true, "mode": "repulse"}, "onclick": {"enable": true, "mode": "push"}, "resize": true},
                    "modes": {"grab": {"distance": 400, "line_linked": {"opacity": 1}}, "bubble": {"distance": 400, "size": 40, "duration": 2, "opacity": 8, "speed": 3}, "repulse": {"distance": 200, "duration": 0.4}, "push": {"particles_nb": 4}, "remove": {"particles_nb": 2}}
                },
                "retina_detect": true
            });
        </script>
        """,
        height=0,
    )

def main():
    particle_background()
    
    st.markdown("<h1 class='main-title'>🎵 Genre-Based Chord Progression Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<div class='glowing-text'>Empowering Musicians with AI-Driven Insights</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://azure.wgp-cdn.co.uk/app-pianist/posts/austin-pacheco-703798-unsplash.jpg" alt="Introduction Image" style="width:100%; height:auto;">
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("Picture Credit: pianistmagazine.com")

    # Introduction/Background
    st.markdown("<div class='section-container blue-gradient'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Introduction/Background</h2>", unsafe_allow_html=True)
    st.write("""
    Our topic is about classifying music genres based on chord progressions. Chord progressions can be a strong indicator of genre [2], and we aim to create a model that classifies a song's genre using its chords.
    """)
    
    st.markdown("<h3 class='subsection-title'>Literature Review</h3>", unsafe_allow_html=True)
    st.write("""
    Prior research has used song lyrics for genre classification [3], while others have explored chord progressions but relied on CNNs and Spotify API data. We plan to focus specifically on chord progressions as our classification metric.
    """)
    
    st.markdown("<h3 class='subsection-title'>Dataset Description</h3>", unsafe_allow_html=True)
    st.write("""
    We have identified a dataset with 135,783 songs, offering details such as artist name, genre, lyrics, and chords, which will help us retrieve the necessary data for our model [4]. Additionally, a MIDI dataset will provide deeper insights into musical features like loudness, pitch, and energy for further analysis [5].
    """)
    
    st.markdown("<h3 class='subsection-title'>Dataset Links</h3>", unsafe_allow_html=True)
    st.markdown("[Chords and Lyrics Dataset](https://www.kaggle.com/datasets/eitanbentora/chords-and-lyrics-dataset)")
    st.markdown("[Lakh MIDI Dataset](https://www.kaggle.com/datasets/imsparsh/lakh-midi-clean/data)")
    st.markdown("</div>", unsafe_allow_html=True)

    # Problem Definition
    st.markdown("<div class='section-container green-gradient'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Problem Definition</h2>", unsafe_allow_html=True)
    st.write("""
    Problem: Musicians, both experienced and emerging, often face challenges when trying to create music that aligns with the vibe or essence of a particular genre. A major challenge is identifying the right chord progressions that fit seamlessly within their desired genre. Additionally, the vast range of musical genres and subgenres have their own stylistic nuances, which makes it even harder to pinpoint which chord progression best reflects the genre.

    Solution: We want to train an algorithm to classify chord progressions to a specific genre to provide artists real-time feedback and help them align their progressions with the vibe they want, streamlining the creative process.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://images.coolwallpapers.me/picsup/3101833-acoustic_acoustic-guitar_audio_bass_black_blur_chords_classic_close-up_dark_depth-of-field_fret_fretboard_guitar_instrument_jazz_modern_music_musical-instrument_nylon_play_rock_sound_string-instr.jpg" alt="Introduction Image" style="width:100%; height:auto;">
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("Picture Credit: images.coolwallpapers.me")

    # Methods
    st.markdown("<div class='section-container purple-gradient'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Methods</h2>", unsafe_allow_html=True)
    
    st.markdown("<h3 class='subsection-title'>Data Preprocessing</h3>", unsafe_allow_html=True)
    st.write("""
        - Chord embeddings: Convert individual chords in a song into dense, numerical vectors that capture harmonic relationships and similarities between chords. [1]
        - Data cleaning: Remove noise from the chord data by removing unnecessary and inconsistent characters such as tabs and any irrelevant labels like 'intro' or 'verse'.
        - Dimensionality Reduction: Filter out unnecessary features to minimize computation and leave only key things like song_name, chords, lyrics, and maybe chords+lyrics.
        - Feature Engineering: Extract chord transitions (e.g., major to minor, tonic to dominant) and create features based on these changes.
    """)
    
    st.markdown("<h3 class='subsection-title'>Machine Learning Algorithms</h3>", unsafe_allow_html=True)
    st.write("""
        - K-Nearest Neighbors (KNN): Classify songs based on their chord embeddings by comparing the proximity of a song's chord vectors to those of songs in the training set.
        - Support Vector Machine (SVM): Classify songs into genres based on chord progressions by finding the optimal hyperplane that separates genres in the feature space.
        - Random Forest: Combine different aspects of chords and create many different trees that focus on different chord progressions common in each genre.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://ak9.picdn.net/shutterstock/videos/3244579/thumb/1.jpg?i10c=img.resize(height:72)" alt="Introduction Image" style="width:100%; height:auto;">
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("Picture Credit: ak9.picdn.net")

    # Results and Discussion
    st.markdown("<div class='section-container orange-gradient'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Results and Discussion</h2>", unsafe_allow_html=True)
    
    st.markdown("<h3 class='subsection-title'>Quantitative Metrics</h3>", unsafe_allow_html=True)
    st.write("""
        - Accuracy: Percentage of accurate genre classifications over the entire dataset/predictions
        - Genre representation: Distribution of predictions among all the different genres
        - Precision: Measure of the model's ability to correctly classify a song into a genre without misclassifying songs from other genres
        - Recall: Measure of the model's ability to identify all songs of a particular genre
    """)
    
    st.markdown("<h3 class='subsection-title'>Project Goals</h3>", unsafe_allow_html=True)
    st.write("""
        - Accuracy: Greater than 70% (lower due to overlap between genres)
        - Genre representation: A fairly uniform distribution, with a difference between the highest guessed and lowest guessed less than 20%
        - Precision and Recall: Greater than or equal to 0.8
    """)
    
    st.markdown("<h3 class='subsection-title'>Expected Results</h3>", unsafe_allow_html=True)
    st.write("""
    - Accuracy should be higher for more distinct genres (jazz, classical) but lower for genres like pop and rock that have similar chords
    - Assuming all goes well, genre representation should be less than 20% difference between highest and lowest guessed genres
    - The model should achieve high precision and recall scores, indicating it avoids incorrectly classifying songs and successfully captures most songs belonging to each genre
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # References
    st.markdown("<div class='section-container red-gradient'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>References</h2>", unsafe_allow_html=True)
    references = [
        "A. Lahnala, et al., \"Chord embeddings: Analyzing what they capture and their role for next chord prediction and artist attribute prediction,\" in Artificial Intelligence in Music, Sound, Art and Design: 10th International Conference, EvoMUSART 2021, Held as Part of EvoStar 2021, Virtual Event, April 7–9, 2021, Proceedings, Springer International Publishing, 2021.",
        "B. Wundervald, W. Zeviani, \"Machine learning and chord based feature engineering for genre prediction in popular Brazilian music,\" arXiv preprint arXiv:1902.03283, 2019.",
        "M. Leszczynski, A. Boonyanit, and A. Dahl, \"Music Genre Classification using Song Lyrics Stanford CS224N Custom Project.\"",
        "Chords and Lyrics Dataset, Kaggle.",
        "Lakh MIDI Dataset, Kaggle."
    ]
    for i, ref in enumerate(references, 1):
        st.markdown(f"<p class='reference-item'>[{i}] {ref}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()