Installation:
1. conda create -n emodialog
2. conda activate emodialog
3. pip install -r requirements.txt

Data preparation:
1. We use BLIP caption generator for captioning images and use them as image input for emotion classification and explanation generation. 

2. Once the captions are generated for the images, we prepare data for training Seq-Seq models like BART, T5. 

3. These files assume the following directory structure: 
	data/
	data/images/ --> For images
	data/train.pkl --> Training data
	data/val.pkl --> Dev data
	data/test.pkl --> Test data

4. Now run the data preparation using: 
	bash data_prep.sh

5. A set of files (for all results in Table - 4) are saved in "data/" folder.
	a. _ques_aft_expl_gen_emo_gen_emo1_emo2_cap1_cap2_conv_ft_gen_cap --> Uses (Image, Emotion, Caption, Dialog)
	b. _ques_aft_expl_gen_emo_gen_emo1_emo2_cap1_cap2_ft_gen_cap --> Uses (Image, Emotion, Caption)
	c. _ques_aft_expl_gen_emo_gen_emo1_emo2_conv_ft_gen_cap --> Uses (Image, Emotion, Dialog)
	d. _ques_aft_expl_gen_emo_gen_cap1_cap2_conv --> Uses (Caption, Dialog)

Running scripts:
1. To train the explanation generation models for questioner run:
	python ques_aft_emo_expla_gen_train.py

2. To evaluate run:
	python ques_aft_emo_expla_gen_eval.py 

3. To train the explanation generation models for answerer run:
	python ans_aft_emo_expla_gen_train.py

4. To evaluate run:
	python ans_aft_emo_expla_gen_eval.py 		