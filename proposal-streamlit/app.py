import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import os

def local_css(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)
    
    with open(file_path, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def local_pic(file_name, caption):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, file_name)
    st.image(file_path, caption=caption)

local_css("style.css")

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
        <div style="text-align: center; font-size: 18px;">
            <a href="https://github.com/thomaschin35/midi2genre/tree/main" target="_blank">
                🌐 GitHub Repository Link
            </a>
        </div>
        <br><br>
        """, 
        unsafe_allow_html=True
    )
    
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
    st.markdown("<h2 class='section-title'>Introduction</h2>", unsafe_allow_html=True)
    st.write("""
    Our topic is about classifying music genres based on chord progressions. Chord progressions can be a strong indicator of genre [2], and we aim to create a model that classifies a song's genre using its chords. In the realm of music theory, certain chords played in succession within a song, such as one of the most famous progressions I-V-VI-IV, are highly characteristic of a pop song for example. The harmony of multiple notes create a chord and form the basis of a song's structure, richness, and depth, all of which are important features in shaping a song's genre.
    """)
    
    st.markdown("<h3 class='subsection-title'>Literature Review</h3>", unsafe_allow_html=True)
    st.write("""
    Prior research has used song lyrics for genre classification [3], while others have explored chord progressions but relied on CNNs and Spotify API data. Wundervald and Zeviani explored this using random forest within Brazilian music and placed heavy emphasis on PCA and the addition/engineering of new features as a comment on future improvements [2]. We plan to focus specifically on chord progressions as our classification metric while extracting integral features for our predictions.
    
    We have identified a dataset with 135,783 songs, offering details such as artist name, genre, lyrics, and chords, which will help us retrieve the necessary data for our model [4]. Additionally, a MIDI dataset will provide deeper insights into musical features like loudness, pitch, and energy for further analysis [5].
    """)
    
    # Methods section
    st.markdown("<div class='section-container purple-gradient'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Methods</h2>", unsafe_allow_html=True)
    
    st.markdown("<h3 class='subsection-title'>Data Processing & Feature Engineering</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://ak9.picdn.net/shutterstock/videos/3244579/thumb/1.jpg?i10c=img.resize(height:72)" alt="Music Image" style="width:100%; height:auto;">
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("Picture Credit: ak9.picdn.net")
    st.write("""
    Our data processing methods included basic data cleaning of the raw chord progressions, and feature engineering methods like bag of words, harmonic intervals, trigrams, and modular frequencies. 
    
    For the <u>**basic data cleaning**</u>, we standardized the song and artist names by removing extra spaces, punctuation, and converting text to lowercase, then processed chord progressions by removing non-chord elements like tab numbers, irrelevant words (e.g., "intro," "verse"), and special characters. This left them in a uniform format, to ease the processing done later.

    We used <u>**bag of words**</u> to capture the presence of specific chord types across progressions. First, we used a predefined vocabulary of chords, including major, minor, augmented, diminished, and suspended variations, to vectorize each chord progression into a sparse binary matrix (1 if present and 0 if not). This matrix indicates whether each chord in the vocabulary appears in each progression. Then, we grouped equivalent chords (e.g., C# and Db, or C and Cmaj) by summing their binary columns, reducing dimensionality and enhancing interpretability. This added 47 columns of features to the dataset in the end.

    To capture the transitions between chords, we encoded sequences of three consecutive chords as <u>**trigrams**</u>. This helped us identify patterns in songs with similar chord progressions. We extracted these sets of three chords from all the chords in the songs and then these trigrams were transformed into feature vectors, resulting in a high-dimensional sparse matrix. To reduce dimensionality, we applied PCA with 50 components to the trigrams matrix in batches to handle memory constraints. 

    We extracted <u>**harmonic intervals**</u> from the chord progressions by encoding the root note of each chord as a semitone value, which represented its pitch position within a 12-tone scale. We then calculated the intervals (in semitones) between consecutive chords in a progression. The resulting intervals were pooled to obtain summary statistics such as mean, maximum, and minimum interval values, which captured the general "movement" or shifts in pitch in a numeric fashion.

    Lastly, we extracted the <u>**modulation frequency**</u> of chord progressions, which is a measure of how often the "key" or dominant root note changes in a song. First, we extracted root notes from each chord in a progression, isolating the primary pitch for each chord symbol. Then, we determined the most frequently occurring root note, or "dominant root," for each section of the progression. As the dominant root changed between sections, we counted each change as a modulation, representing a shift in musical key. The final modulation frequency for each progression was stored as a feature for each song.
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='subsection-title'>Random Forest</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://mlarchive.com/wp-content/uploads/2022/09/random-forests.jpg" alt="Introduction Image" style="width:100%; height:auto;">
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("Picture Credit: mlarchive.com")
    st.write("""
    Using our final feature set, the first machine learning algorithm we implemented was a random forest. Musical features have many complex relationships that would go into predicting a specific genre. The ensemble of decision trees provided by random forest allows it to handle that complexity and form non-linear relationships between features. We decided to utilize this model also because of its ability to prevent overfitting. Overfitting is a large concern especially when dealing with a vast amount of intricate features and data points. Random forest averages out individual tree predictions which can significantly reduce overfitting, thus helping with the efficacy of our model. Another important reason for using Random forest is its ability to provide the strength of each feature when making a decision, allowing us to easily identify patterns and features to look out for [6]. 
    """)
    
    st.markdown("<h3 class='subsection-title'>Support Vector Machine</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://hands-on.cloud/wp-content/uploads/2021/12/Overview-of-supervised-learning-SVM-1024x576.png" alt="Introduction Image" style="width:100%; height:auto;">
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("Picture Credit: hands-on.cloud")
    st.write("""
    The second machine learning algorithm we decided to implement is a Support Vector Machine (SVM), once again using our final feature set. Some of the processing we did to take the chord progressions and turn it into features resulted in a rather large feature matrix (currently at about 101 features) and this was after using the PCA algorithm on a couple of the heftier features. Because of this larger feature set, we needed an algorithm that would be able to handle this and SVM is thus a great choice. SVM is able to handle all these features (and large data sets in general) on lower computational power due to its memory efficiency [7]. This memory efficiency comes from the model using support vectors that are formed from a subset of the input data and used in the decision function. This algorithm is also useful due to the kernel functions we can modify: similar to the random forest, if the relationship between our features and the genre classifications is non-linear, the kernel can transform the data into a hyperplane, allowing for more accurate classification and maintaining a linear divide in the data. We chose to use the RBF kernel for our training due to the non-linearity nature of our data.
    """)

    st.markdown("<h3 class='subsection-title'>K-Nearest Neighbors</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://miro.medium.com/v2/resize:fit:640/format:webp/0*2_qzcm2gSe9l67aI.png" alt="Introduction Image" style="width:100%; height:auto;">
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("Picture Credit: researchgate.net")
    st.write("""
    The third machine learning algorithm we decided to implement is a K-Nearest-Neighbors algorithm, using the final feature set like our other algorithms. KNN fits our problem because it doesn’t assume any type of linear relationship between the data and the output, allowing for flexibility within the processing, be it linear or complex patterns. KNN also inherently deals with multiclass classification, something we clearly need to distinguish between multiple different music genres. Our dataset also implies some similarity between the pop and rock chord progressions, with fuzzy differences to the naked eye. However, KNN can adapt to these and deal with complicated decision boundaries. It does this by using proximity metrics. It takes the closest neighbors in the feature space and gives more weight to those data points to accurately classify and cluster the data even if there’s overlap. All these reasons make KNN an effective choice for our specific problem.
    """)
    
    # Results and Discussion section
    st.markdown("<div class='section-container orange-gradient'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Results and Discussion</h2>", unsafe_allow_html=True)
    
    st.markdown("<h3 class='subsection-title'>Random Forest Results</h3>", unsafe_allow_html=True)
    
    local_pic("ConfusionMatrixRF.png", caption="Figure 1: Confusion Matrix for Random Forest")

    st.write("""
    **Strong prediction for overrepresented classes:**

    The model performed the best in recognizing Pop, Rock, and World which dominate the dataset. These genres have the highest number of correct predictions on the diagonal of the confusion matrix, where the predicted class matches the true class:

    - Pop: 3,823 samples were correctly classified.
    - Rock: 3,410 samples were correctly classified.
    - World: 2,977 samples were correctly classified.

    This would indicate that the model has learned to recognize features associated with these majority genres. However the matrix also shows strong genre confusion.

    **Genre Confusion for overrepresented classes:**

    Despite their high number of correct predictions, Pop, Rock, and World also had high misclassification rates, particularly as each other. In the confusion matrix, the shading highlights where Rock was confused with Pop and World with both Pop and Rock. This suggests that, even with more data, the model struggles to differentiate these genres, likely due to overlapping features. The lack of distinction between these genres, despite their high representation, indicates similar or insufficiently unique features that lead to frequent misclassifications.
    """)
    
    rf_quant_metrics = {
        "Genre": ["Classical", "Country", "Electronic", "Folk", "Hiphop", "Jazz", "Pop", "Rock", "Soul", "World"],
        "Precision": [1.0, 0.38, 1.0, 0.29, 0.7, 0.54, 0.4, 0.4, 0.34, 0.43],
        "Recall": [0.01, 0.18, 0.0, 0.0, 0.02, 0.53, 0.53, 0.48, 0.03, 0.49],
        "F1-Score": [0.03, 0.25, 0.01, 0.01, 0.04, 0.54, 0.45, 0.44, 0.06, 0.46],
        "Support": [67, 2035, 255, 1512, 672, 896, 7223, 7035, 1408, 6040]
    }
    st.table(pd.DataFrame(rf_quant_metrics))
    st.markdown(
        """
        <style>
        .dataframe {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<p style='text-align: center; font-size: 0.9em; color: gray;'>Figure 2: Qualitative Metrics for Random Forest Model</p>", unsafe_allow_html=True)
    
    st.write("""
    This is further represented by the qualitative metrics (Shown in Figure 2)
    - Precision
    - Recall
    - F1-Score
    - Support
    
    The F1-score, which balances precision and recall, is essential for evaluating this imbalanced dataset, as it highlights the model’s effectiveness across both frequent and rare genres. Higher F1-scores for Pop (0.45), Rock (0.44), and World (0.46) indicate that the model is more effective at predicting these classes, likely due to their overrepresentation. This allows the model to learn nuanced patterns, leading to higher confidence in predictions.

    In contrast, underrepresented genres like Classical, Electronic, and Folk have extremely low F1-scores (0.03, 0.01, and 0.01, respectively), showing the model’s struggle with these classes. Low recall for these genres, such as Classical and Electronic (0.01 and 0.00), suggests the model rarely identifies these classes, reflecting an inability to generalize to minority classes. High precision but near-zero recall for these genres implies the model predicts them correctly when it does, but rarely makes those predictions, impacting the F1-scores.
    """)
    
    st.markdown("<h3 class='subsection-title'>SVM Results</h3>", unsafe_allow_html=True)
    
    svm_quant_metrics = {
        "Genre": ["Classical", "Country", "Electronic", "Folk", "Hiphop", "Jazz", "Pop", "Rock", "Soul", "World"],
        "Precision": [0.00, 0.38, 0.00, 0.00, 0.00, 0.54, 0.39, 0.38, 0.62, 0.43],
        "Recall": [0.00, 0.07, 0.00, 0.00, 0.00, 0.61, 0.56, 0.50, 0.01, 0.43],
        "F1-Score": [0.00, 0.12, 0.00, 0.00, 0.00, 0.57, 0.46, 0.44, 0.02, 0.43],
        "Support": [71, 1916, 271, 1416, 634, 922, 7179, 7060, 1430, 6049]
    }
    st.table(pd.DataFrame(svm_quant_metrics))
    st.markdown(
        """
        <style>
        .dataframe {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<p style='text-align: center; font-size: 0.9em; color: gray;'>Figure 3: Qualitative Metrics for SVM Model</p>", unsafe_allow_html=True)
    
    st.write("""
    The SVM model demonstrates inconsistent performance across genres, with Jazz being the strongest performer (f1-score: 0.57) and complete failure in classifying Classical, Electronic, Folk, and Hiphop (all metrics at 0.00). Pop and Rock show moderate performance (f1-scores of 0.46 and 0.44 respectively), benefiting from large sample sizes (7179 and 7060 samples).

    Despite high precision in some cases, like Soul (0.62), extremely low recall values lead to poor overall performance, suggesting the model struggles with consistent genre identification. The significant class imbalance in the dataset, ranging from 71 samples (Classical) to 7179 samples (Pop), likely contributes to these varying results.
    """)
    
    local_pic("ConfusionMatrixSVM.png", caption="Figure 4: Confusion Matrix for SVM")
    
    st.write("""
    The confusion matrix reveals significant classification challenges in the model's performance. Pop and Rock show the highest number of correct predictions (4006 and 3551 respectively), but there's substantial confusion between these two genres, with 1821 Pop songs misclassified as Rock and 2345 Rock songs misclassified as Pop. World music also shows a decent number of correct predictions (2624), but experiences considerable confusion with Pop and Rock genres. Most notably, Classical, Electronic, Folk, and Hiphop genres show extremely poor performance with almost no correct predictions, with their samples being predominantly misclassified as either Pop or Rock. This pattern suggests the model has a strong bias toward the two dominant classes (Pop and Rock), likely due to their larger representation in the training data.
    """)

    st.markdown("<h3 class='subsection-title'>KNN Results</h3>", unsafe_allow_html=True)
    local_pic("ConfusionMatrixKNN.png", caption="Figure 5: Confusion Matrix for KNN")

    st.write("""The KNN model performed the best in recognizing Jazz, Pop, Rock, and World, which were the most occurring in the dataset. These genres have the highest number of correct predictions as shown in the confusion matrix, where the predicted class matches the true class. In terms of correct classifications, Pop had 3,942, Rock had 2,987, and World had 2,996. This shows that the model learned to recognize features associated these genres the best. Unfortunately, the matrix also shows a lot of confusion between the genres, as the shading around misclassifications surrounding those genres is darker.
    """)

    st.write("""Pop, Rock, and World genres were often misclassified for one another. In the confusion matrix, the shading highlights where Rock was confused with Pop and vice versa and World was confused with both Pop and Rock. This implies that the model struggles to differentiate these genres, most likely due to overlapping features, since the genres use similar chord progressions and styles. The lack of distinction between these genres, despite their high representation, likely lead to features that were also non-distinct, making it harder for the model to differentiate them.
    """)

    knn_quant_metrics = {
        "Genre": ["Classical", "Country", "Electronic", "Folk", "Hiphop", "Jazz", "Pop", "Rock", "Soul", "World"],
        "Precision": [0.14, 0.33, 0.20, 0.22, 0.32, 0.56, 0.39, 0.42, 0.37, 0.41],
        "Recall": [0.01, 0.22, 0.00, 0.01, 0.01, 0.53, 0.55, 0.42, 0.05, 0.50],
        "F1-Score": [0.03, 0.26, 0.01, 0.02, 0.02, 0.55, 0.46, 0.42, 0.08, 0.45],
        "Support": [71, 1916, 271, 1416, 634, 922, 7179, 7060, 1430, 6049]
    }
    st.table(pd.DataFrame(knn_quant_metrics))
    st.markdown(
        """
        <style>
        .dataframe {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<p style='text-align: center; font-size: 0.9em; color: gray;'>Figure 6: Qualitative Metrics for KNN Model</p>", unsafe_allow_html=True)

    st.write("""The F1-score is a good representation of both the precision and recall the model has for each genre. Jazz(0.55), Pop (0.46), World (0.45), and Rock (0.42) all had higher F1-scores which suggests that the model is more effective at predicting these classes, likely due to their overrepresentation and also Jazz’s distinct nature musically. This allowed the model to learn more subtle pattern differences, leading to higher confidence in predictions.
             \nWhen looking at genres like Soul, Classical, Folk, and Electronic we see extremely low F1-scores (0.08, 0.03, 0.02, and 0.01, respectively), showing the model’s struggle with these classes likely due to how underrepresented they are in the dataset. Low recall for these genres, such as Classical, Folk, Hip-Hop, and Electronic (0.01, 0.01, 0.01, and 0.00), also highlights that the model rarely identifies these classes, exposing the model’s inability to notice clear patterns for minority genres. There was also relatively high precision but near-zero recall for these genres which reveals that the model predicts them fairly correctly when it does, but rarely makes those predictions due to their sparseness in the dataset, impacting the F1-scores. It's clear the KNN model does better with the genres we had more to train on, and still requires some more advanced feature engineering to distinguish between them.
    """)

    st.markdown("<h3 class='subsection-title'>Looking at All Models by F1 Scores</h3>", unsafe_allow_html=True)
    all_summary_metrics = {
        "Genre": ["Classical", "Country", "Electronic", "Folk", "Hip-Hop", "Jazz", "Pop", "Rock", "Soul", "World"],
        "KNN": [0.03, 0.26, 0.01, 0.02, 0.02, 0.55, 0.46, 0.42, 0.08, 0.45],
        "Random Forest": [0.03, 0.26, 0.01, 0.02, 0.02, 0.55, 0.46, 0.42, 0.08, 0.45],
        "SVM": [0.00, 0.12, 0.00, 0.00, 0.00, 0.57, 0.46, 0.44, 0.02, 0.43]
    }
    st.table(pd.DataFrame(all_summary_metrics))
    st.markdown(
        """
        <style>
        .dataframe {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<p style='text-align: center; font-size: 0.9em; color: gray;'>Figure 7: F1 Scores for Three Models Trained</p>", unsafe_allow_html=True)

    st.write("""This project evaluated three machine learning models—K-Nearest Neighbors (KNN), Random Forest, and Support Vector Machine (SVM)—for classifying musical genres based on chord progressions. KNN performed best, achieving higher F1-scores for genres like Jazz, Pop, Rock, and World due to its adaptability to complex, non-linear feature distributions. However, it struggled with underrepresented genres like Classical and Electronic, reflecting its sensitivity to class imbalances. Random Forest, while effective for majority genres and strong in feature interpretation, suffered from significant genre confusion, particularly between overlapping genres such as Pop and Rock. SVM, despite its computational efficiency with an RBF kernel, performed poorly overall, especially for minority genres, due to its inability to handle class imbalances and subtle feature variations.
    \nConfusion matrices revealed consistent difficulties in distinguishing between Pop, Rock, and World genres across all models. Genre confusion was most pronounced in Random Forest, while KNN showed better adaptability but remained constrained by dataset imbalances. SVM’s reliance on sparse data representations resulted in near-zero recall for underrepresented genres. Overall, KNN proved the strongest for majority genres, though its limitations, along with those of the other models, underscore the need for enhanced data balancing and feature engineering. Future improvements could include synthetic data generation, exploring additional harmonic and rhythmic features, and integrating deep learning models like LSTMs to better capture sequential patterns in chord progressions.""")
    st.markdown("<h3 class='subsection-title'>Further Analysis</h3>", unsafe_allow_html=True)
    local_pic("features-pca.png", caption="Figure 8: 2D PCA Plot of Our Features")
    st.write("""
    We performed Principal Component Analysis (PCA) on the dataset to examine the variability and potential genre clusters based on our features. The results indicate a lack of variance in the principal components, suggesting that the features do not effectively distinguish genres.

    We experimented with several techniques to improve model performance:
    - Class Weights: Adjusted to give more importance to underrepresented genres
    - Subset Sampling: Created balanced subsets by sampling equal numbers for each genre
    - SMOTE: Applied to synthetically balance the dataset

    Despite these efforts, each approach yielded an accuracy around 40%, with SMOTE actually lowering accuracy further.
    """)

    st.markdown("<div class='section-container purple-gradient'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Future Work</h2>", unsafe_allow_html=True)

    st.write("""Although we were able to do some work on engineering the features of the engineering model, for the timing of this project we were not able to fully implement the chord embedding model. In the future, we would like to delve into how this different representation of chords will impact the accuracy of the 3 models for classifying genres.""")
    st.markdown("<h3 class='subsection-title'>Chord Embeddings as Language</h3>", unsafe_allow_html=True)
    local_pic("lstm-architecture.webp", caption="Figure 9: LSTM Architecture that could be used to train for genre classification")

    st.write("""Chord embeddings can be treated like a language due to their sequential structure, similar to how words form sentences. Transformer models, such as those used in symbolic music generation, can be used for processing sequential data like chord progressions. By leveraging self-attention mechanisms, these models can capture long-term dependencies and context within sequences. For example, pretrained transformer models like BERT or GPT can be fine-tuned on chord sequences to learn relationships specific to musical genres, building on techniques shown to enhance symbolic music processing [8][9].
    \nFor the sake of simplicity, models like Word2Vec or FastText could create embeddings based on harmonic relationships, although these may lack the contextual depth of transformers. Alternatively, recurrent neural networks (RNNs) like LSTMs or GRUs can account for temporal dependencies in chord sequences, capturing the influence of earlier chords on later ones [9].
    """)

    st.markdown("<h3 class='subsection-title'>Chord Embeddings as Graphs</h3>", unsafe_allow_html=True)
    local_pic("chord-graphs.jpg", caption="Figure 10: A possible diagram of treating chord as graphs")
    st.write("""Another approach would involve representing chord embeddings as graphs, where nodes represent unique chords and edges represent transitions between chords, weighted by their frequency or context. Graph-based approaches have been successfully used in tasks like melody-to-chord harmonization, where structural patterns in music are important. For example, integrating Graph Neural Networks (GNNs) to learn chord relationships could help model genre-specific motifs such as secondary dominants in jazz or cyclic structures in pop[9]. GNNs can be enhanced by incorporating additional features, such as edge Directionality (capturing the difference between A → D major and D → A major transitions) and embedding nodes with harmonic intervals and frequency distributions.
    \nGraph-based embeddings, combined with sequence models like transformers, could offer a hybrid solution to address both structural and temporal aspects of chord progressions, making this a promising area for further research [8][9].
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # References section
    st.markdown("<div class='section-container red-gradient'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>References</h2>", unsafe_allow_html=True)
    references = [
        "A. Lahnala, et al., \"Chord embeddings: Analyzing what they capture and their role for next chord prediction and artist attribute prediction,\" in Artificial Intelligence in Music, Sound, Art and Design: 10th International Conference, EvoMUSART 2021.",
        "B. Wundervald, W. Zeviani, \"Machine learning and chord based feature engineering for genre prediction in popular Brazilian music,\" arXiv preprint arXiv:1902.03283, 2019.",
        "M. Leszczynski, A. Boonyanit, and A. Dahl, \"Music Genre Classification using Song Lyrics Stanford CS224N Custom Project.\"",
        "Chords and Lyrics Dataset, Kaggle.",
        "Lakh MIDI Dataset, Kaggle.",
        "AIML.com. (2024, May 22). What are the advantages and disadvantages of Random Forest?",
        "Scikit-Learn Developers. (n.d.). Support vector machines. Scikit-Learn.",
        "N. Zhang, \"Learning Adversarial Transformer for Symbolic Music Generation,\" in IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 4, pp. 1754-1763, April 2023, doi: 10.1109/TNNLS.2020.2990746.",
        "S. Rhyu, H. Choi, S. Kim and K. Lee, \"Translating Melody to Chord: Structured and Flexible Harmonization of Melody With Transformer,\" in IEEE Access, vol. 10, pp. 28261-28273, 2022, doi: 10.1109/ACCESS.2022.3155467"
    ]
    for i, ref in enumerate(references, 1):
        st.markdown(f"<p class='reference-item'>[{i}] {ref}</p>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://images.coolwallpapers.me/picsup/3101833-acoustic_acoustic-guitar_audio_bass_black_blur_chords_classic_close-up_dark_depth-of-field_fret_fretboard_guitar_instrument_jazz_modern_music_musical-instrument_nylon_play_rock_sound_string-instr.jpg" alt="Guitar Image" style="width:100%; height:auto;">
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("Picture Credit: images.coolwallpapers.me")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-container green-gradient'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Project Management</h2>", unsafe_allow_html=True)

    local_pic("ml-14-midterm-gantt-chart.png", caption="Figure 11: Gantt Chart used for this project")
    
    contributions_table = {
        "Name": ["Thomas", "Hannah", "Kasikrit", "Michaelah", "Tooni"],
        "Contribution": [
            "Gantt Chart, Data preprocessing/feature engineering, Github file directory, Video",
            "Methods - KNN, Video",
            "Website Creation using stremlit, Comparison between all the models",
            "Future work",
            "Data Processing Methods KNN, Video"
        ]
    }
    st.table(pd.DataFrame(contributions_table))
    st.markdown(
        """
        <style>
        .dataframe {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<p style='text-align: center; font-size: 0.9em; color: gray;'>Figure 12: Contribution Table of the Team</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()