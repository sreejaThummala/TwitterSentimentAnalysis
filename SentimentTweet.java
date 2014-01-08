import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Vector;

import javax.rmi.CORBA.Util;

import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.attribute.StringToWordVector;


public class SentimentTweet{
	
	// config params:
	String jwnlFilePath;
	String openNlpModelsPath;
	String trainFilePath;
	String testFilePath;
	String trainFolder;
	String testFolder;
	String emoticonsFilePath;
	String outputFile;
	HashSet<String> emoticons = new HashSet<String>();
	
	private String featuresSetStr;
	
	private NLP nlp;
	private SWNUtil util = new SWNUtil();
	
	
	enum options {
		FP,
		FF,
		TFIDF
	};

	
	public SentimentTweet(String configFile) throws Exception {
		loadConfig(configFile);
		System.out.println("config file loaded");
		BufferedReader in = new BufferedReader(new FileReader(emoticonsFilePath));
		String line = in.readLine();
		System.out.println("reading emoticons");
		while (line != null) {
			emoticons.add(line);
			line = in.readLine();
		}
		in.close();
		System.out.println("completed reading emoticons");
	
		System.out.println("Initializing nlp");
		nlp = new NLP(openNlpModelsPath, jwnlFilePath);
		System.out.println("Initialization complete");
		Vector<Entry> data = null;
		
		System.out.println("reading the trainingData");
		data = readTwitterContent(trainFilePath,0);
		System.out.println("completed reading trainig Data");
		
		System.out.println("cross validation set creation");
		//Vector<CVPair> cvpairs = buildCV(data, 10);
		System.out.println("creation completed");
		
		/*
		 * arguments for weighting functions
		 * 0 for FP
		 * 1 for FF
		 * 2 for TFIDF
		 * Work or not to work!!!!!!!!!
		 */
		System.out.println("+++++++++++++++++++++++++++++++++");
		System.out.println("run and test all features");
		
	
			calculateFeatures(data,features.POS_COUNT,false);
			Vector<Stat> averageStats = new Vector<Stat>();
			
			//for (int i = 0; i < cvpairs.size(); ++i) {
				
				Vector<Stat> stats = runTrainAndTest(data, data, options.FP);
				/*for (int j = 0; j < stats.size(); ++j) {
					if (i == 0)
						averageStats.add(stats.get(j));
					else
						averageStats.get(j).applyToAverage(stats.get(j), i + 1);
				}*/
		//	}
				calculateFeatures(data,features.AVGLASTPOL,false);
			averageStats = new Vector<Stat>();
				/*
				//for (int i = 0; i < cvpairs.size(); ++i) {
					
					 stats = runTrainAndTest(data, data, options.FP);
					 calculateFeatures(data,features.WORDTOKENS,false);
						averageStats = new Vector<Stat>();
							
							//for (int i = 0; i < cvpairs.size(); ++i) {
								
								 stats = runTrainAndTest(data, data, options.FP);
							 calculateFeatures(data,features.SPECIALWORDS,false);
									averageStats = new Vector<Stat>();
										
										//for (int i = 0; i < cvpairs.size(); ++i) {
											
											 stats = runTrainAndTest(data, data, options.FP);
											 calculateFeatures(data,features.EMOTICONNUMBER,false);
												averageStats = new Vector<Stat>();*/
													
													//for (int i = 0; i < cvpairs.size(); ++i) {
														
														 stats = runTrainAndTest(data, data, options.FP);
			System.out.println("Feature set " + features.WORD);
			System.out.println("=======================");
			System.out.println("Option"+"FF");
			for (int j = 0; j < averageStats.size(); ++j) {
				System.out.println(averageStats.get(j).toString());
			}
			System.out.println("");
			System.out.flush();
		
		//}
		
	}
	
	/*
	 * running and testing all the classifiers on a specific split from the 
	 * cross validation set.
	 * stringToWordOptions determines the weighting function to use.
	 */
	Vector<Stat> runTrainAndTest(Vector<Entry> train, Vector<Entry> test, options option) throws Exception {
		createDataFolder(train, trainFolder);
		
		TextDirectoryLoader loader = new TextDirectoryLoader();
	    loader.setDirectory(new File(trainFolder));
	    Instances dataRaw = loader.getDataSet();
	    StringToWordVector filter = new StringToWordVector();
	    String stringToWordOptions = convertStringToWord(option);
	    //filter.setOptions(weka.core.Utils.splitOptions(""));
	    filter.setInputFormat(dataRaw);
	    Instances trainFiltered = Filter.useFilter(dataRaw, filter);
	    Reorder reorder = new Reorder();
		//reorder.setOptions(weka.core.Utils.splitOptions("-R 2-last,first"));
		reorder.setInputFormat(trainFiltered);
		trainFiltered = Filter.useFilter(trainFiltered, reorder);
		
		createDataFolder(test, testFolder);
		loader.setDirectory(new File(testFolder));
	    dataRaw = loader.getDataSet();
	    //filter.setInputFormat(dataRaw);
	    Instances testFiltered = Filter.useFilter(dataRaw, filter);
	    reorder.setInputFormat(testFiltered);
		testFiltered = Filter.useFilter(testFiltered, reorder);
		
		Vector<Stat> stats = new Vector<Stat>();
		
	/*	System.out.println("NAIVE BAYES - multinomial");
		SMO classifier = new LibSVM();

		String[] options = weka.core.Utils.splitOptions("-Q -P 100 -S 1 -I 100 ");
		AdaBoostM1 tree = new AdaBoostM1();
		tree.setClassifier( classifier );
		tree.setOptions(options);
		tree.buildClassifier(trainFiltered);
		Evaluation eval=new Evaluation(trainFiltered);
		eval.crossValidateModel(tree,trainFiltered,10,new Random(1));
		System.out.println(eval.toSummaryString("\n Results\n========\n",true));*/
	/*	
	Classifier classifier = new NaiveBayesMultinomial();
	String []options = weka.core.Utils.splitOptions("");
		classifier.setOptions(options);
		Evaluation eval = new Evaluation(trainFiltered);
		eval.crossValidateModel(classifier, trainFiltered, 10, new Random(1));
		System.out.println(eval.toSummaryString());
		System.out.println(eval.recall(0));
		System.out.println(eval.precision(0));
		System.out.println(eval.fMeasure(1));
		System.out.println(eval.fMeasure(2));
		/*Stat stat = testClassifier(classifier, trainFiltered, testFiltered);
		
		stat.classifierName = "NaiveBayes";
		stats.add(stat);*/
		
		Evaluation eval = new Evaluation(trainFiltered);
		System.out.println("SVM with polynimial degree 2 kernel");
		Classifier classifier = new LibSVM();
		String[] options = weka.core.Utils.splitOptions("-K 0");
		classifier.setOptions(options);
		eval.crossValidateModel(classifier, trainFiltered, 10, new Random(1));
		System.out.println(eval.toSummaryString());
		/*stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "LibSVM-POLY2";
		stats.add(stat);*/
		/*
		System.out.println("SVM with linear kernel");
		 classifier = new LibSVM();
		options = weka.core.Utils.splitOptions(" -K 0 ");
		classifier.setOptions(options);
		 eval = new Evaluation(trainFiltered);
		eval.crossValidateModel(classifier, trainFiltered, 10, new Random(1));
		System.out.println(eval.toSummaryString());
		/*
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "LibSVM-LIN";
		stats.add(stat);
		/*
		System.out.println("kNN - no normalization (k = 1)");
		classifier = new IBk();
		options = weka.core.Utils.splitOptions("-K 1 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
		classifier.setOptions(options);
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "KNN(1)";
		stats.add(stat);*/
		
		/*/// kNN - no normalization (k = 3)
		classifier = new IBk();
		options = weka.core.Utils.splitOptions("-K 3 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
		classifier.setOptions(options);
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "KNN(3)";
		stats.add(stat);
		
		/// kNN - no normalization (k = 5)
		classifier = new IBk();
		options = weka.core.Utils.splitOptions("-K 5 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
		classifier.setOptions(options);
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "KNN(5)";
		stats.add(stat);
		
		/// kNN - no normalization (k = 10)
		classifier = new IBk();
		options = weka.core.Utils.splitOptions("-K 10 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
		classifier.setOptions(options);
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "KNN(10)";
		stats.add(stat);
		
		/// kNN - no normalization (k = 20)
		classifier = new IBk();
		options = weka.core.Utils.splitOptions("-K 20 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
		classifier.setOptions(options);
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "KNN(20)";
		stats.add(stat);
		
		/// kNN - no normalization (k = 30)
		classifier = new IBk();
		options = weka.core.Utils.splitOptions("-K 30 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
		classifier.setOptions(options);
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "KNN(30)";
		stats.add(stat);
		
		/// kNN - no normalization (k = 50)
		classifier = new IBk();
		options = weka.core.Utils.splitOptions("-K 50 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance -D\\\"\"");
		classifier.setOptions(options);
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "KNN(50)";
		stats.add(stat);
		
		/// kNN - with normalization (k = 1)
		classifier = new IBk();
		options = weka.core.Utils.splitOptions("-K 1 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance\\\"\"");
		classifier.setOptions(options);
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "KNN(1)Norm";
		stats.add(stat);
		
		/// kNN - with normalization (k = 3)
		classifier = new IBk();
		options = weka.core.Utils.splitOptions("-K 3 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance\\\"\"");
		classifier.setOptions(options);
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "KNN(3)Norm";
		stats.add(stat);
		
		/// kNN - with normalization (k = 5)
		classifier = new IBk();
		options = weka.core.Utils.splitOptions("-K 5 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance\\\"\"");
		classifier.setOptions(options);
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "KNN(5)Norm";
		stats.add(stat);
		
		/// kNN - with normalization (k = 10)
		classifier = new IBk();
		options = weka.core.Utils.splitOptions("-K 10 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance\\\"\"");
		classifier.setOptions(options);
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "KNN(10)Norm";
		stats.add(stat);
		
		/// kNN - with normalization (k = 20)
		classifier = new IBk();
		options = weka.core.Utils.splitOptions("-K 20 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance\\\"\"");
		classifier.setOptions(options);
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "KNN(20)Norm";
		stats.add(stat);
		
		/// kNN - with normalization (k = 30)
		classifier = new IBk();
		options = weka.core.Utils.splitOptions("-K 30 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance\\\"\"");
		classifier.setOptions(options);
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "KNN(30)Norm";
		stats.add(stat);
		*//*
		System.out.println("KNN(1) norm ");
		/// kNN - with normalization (k = 50)
		classifier = new IBk();
		options = weka.core.Utils.splitOptions("-K 50 -A \"weka.core.neighboursearch.LinearNNSearch -A \\\"weka.core.EuclideanDistance\\\"\"");
		classifier.setOptions(options);
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName = "KNN(50)Norm";
		stats.add(stat);
	*/
		System.out.println("max entropy :Logistic regression");
		//classifier = new IBk();/
		/*
		Logistic classifier = new weka.classifiers.functions.Logistic();
		classifier.setOptions(weka.core.Utils.splitOptions("-R 1.0E-8 -M -1"));
		//classifier.setOptions(options);
		Evaluation eval = new Evaluation(trainFiltered);
		eval.crossValidateModel(classifier, trainFiltered, 10, new Random(1));
		System.out.println(eval.toSummaryString());
		/*
		eval = new Evaluation(trainFiltered);
		eval.crossValidateModel(classifier, trainFiltered, 10, new Random(1));
		System.out.println(eval.toSummaryString());
		
		stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName="MaxEnt(Logistic Regression)";
		stats.add(stat);
		*/
	/*
		System.out.println("max entropy: SMO");
		SMO classifier = new weka.classifiers.functions.SMO();
		classifier.setOptions(weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));
		String []options = weka.core.Utils.splitOptions("");
		Evaluation eval = new Evaluation(trainFiltered);
		eval.crossValidateModel(classifier, trainFiltered, 10, new Random(1));
		System.out.println(eval.toSummaryString());
		System.out.println(eval.fMeasure(0));
		
		/*stat = testClassifier(classifier, trainFiltered, testFiltered);
		stat.classifierName="MaxEnt(SMO)";
		stats.add(stat);*/
		return stats;
	}
	
	private String convertStringToWord(options option) {
		String optionString =null;
		switch(option){
		case FP:
			optionString = "";
			
		case FF:
			optionString = "-T";
			
		case TFIDF:
			optionString = "-I -T";
		}
		return optionString;
	}

	/*
	 * build a model and test a specific classifier on a given testset.
	 * returns the accuracy running on the test and train sets.
	 */
	Stat testClassifier(Classifier classifier, Instances trainData, Instances testData) throws Exception {
		Stat stat = new Stat();
		classifier.buildClassifier(trainData);
		double accuracy = 0;
		for (int i = 0; i < testData.numInstances(); ++i) {
			Instance inst = testData.instance(i);
			
			double pred = classifier.classifyInstance(inst);
			if (inst.classValue()==pred) {
				accuracy++;
			} 
		}
		accuracy = accuracy / (double)testData.numInstances();
		System.out.println(accuracy+" test");
		stat.accuracyTest = accuracy;
		
		accuracy = 0;
		for (int i = 0; i < trainData.numInstances(); ++i) {
			Instance inst = trainData.instance(i);
			double pred = classifier.classifyInstance(inst);
			if (inst.classValue() == pred)
				accuracy++;
		}
		accuracy = accuracy / (double)trainData.numInstances();
		System.out.println(accuracy+" train");
		stat.accuracyTrain = accuracy;
		
		return stat;
	}
	
	/*
	 * creating a cross-validation set
	 */
	Vector<CVPair> buildCV(Vector<Entry> data, int fold) throws Exception {
		Vector<CVPair> pairs = new Vector<CVPair>();
		int amountInSet =  data.size() / fold;
		Vector<Integer> indices = new Vector<Integer>();
		for (int i = 0; i < data.size(); ++i)
			indices.add(new Integer(i));
		for (int i = 0; i < fold; ++i) {
			CVPair cvpair = new CVPair();
			pairs.add(cvpair);
			while (cvpair.test.size() < amountInSet) {
				int rand = (int)(Math.random() * indices.size());
				Entry entry = data.get(indices.get(rand).intValue());
				cvpair.test.add(entry);
				indices.remove(rand);
			}
		}
		
		for (int i = 0; i < pairs.size(); ++i) {
			CVPair p = pairs.get(i);
			for (int j = 0; j < pairs.size(); ++j) {
				if (j == i)
					continue;
				p.train.addAll(pairs.get(j).test);
			}
		}
		return pairs;
	}
	
	/*
	 * preparing a data folder to be be read by the StringToWords filter of WEKA
	 */
	void createDataFolder(Vector<Entry> data, String folder) throws IOException {
		File mainFolder = new File(folder);
		if (mainFolder.exists()) {
			removeDirectory(mainFolder);
			mainFolder.mkdir();
		}
		// create the train folder
		for (int i = 0; i < data.size(); ++i) {
			File classFolder = new File(folder + "/class" + data.get(i).label);
			if (!classFolder.exists())
				classFolder.mkdir();
			PrintWriter pw = new PrintWriter(folder + "/class" + data.get(i).label + "/twit" + i + ".txt");
			
			Iterator<String> iter = data.get(i).features.keySet().iterator();
			while (iter.hasNext()) {
				String word = iter.next();
				Integer num = data.get(i).features.get(word);
				for (int j = 0; j < num.intValue(); ++j)
					pw.print(word + " ");
			}
			pw.close();
		}
	}
	
	/*
	 * loading configuration parameters from the configuration file
	 */
	private void loadConfig(String configFile) throws Exception {
		BufferedReader in = new BufferedReader(new FileReader(configFile));
		String line = in.readLine();
		
		while (line != null) {
			// param=value
			
			String []tokens = line.split("[=]");
			if (tokens[0].equals("JWNL_FILE_PATH")) {
				jwnlFilePath = tokens[1];
			} else if (tokens[0].equals("OPEN_NLP_MODELS_FOLDER")) {
				openNlpModelsPath = tokens[1];
			} else if (tokens[0].equals("TRAIN_FILE_PATH")) {
				trainFilePath = tokens[1];
			} else if (tokens[0].equals("TEST_FILE_PATH")) {
				testFilePath = tokens[1];
			}  else if (tokens[0].equals("TRAIN_FOLDER")) {
				trainFolder = tokens[1];
			} else if (tokens[0].equals("TEST_FOLDER")) {
				testFolder = tokens[1];
			} else if (tokens[0].equals("EMOTICONS_FILE_PATH")) {
				emoticonsFilePath = tokens[1];
			} else if (tokens[0].equals("OUTPUT_FILE")) {
				outputFile = tokens[1];
			}
			line = in.readLine();
		}
		in.close();
	}
	
	
	
	/*
	 * reading Twitter data.
	 * Pre-process to match the training Data model
	 * The format of the training Data and test data is completely different 
	 * So need different readers to infer about the tweets and the result
	 * mode : 0 - training data
	 * mode : 1 - test data
	 */
	private Vector<Entry> readTwitterContent(String dataFile, int mode) throws Exception {
		Vector<Entry> out = new Vector<Entry>();
		BufferedReader in = new BufferedReader(new FileReader(dataFile));
		int counter = 0;
		String line = in.readLine();
		int posCounter = 0;
		int negCounter = 0;
		while (line != null) {
			counter++;
			if (counter % 100 == 0)
				System.out.println("" + counter + " lines were loaded");
			String []tokens = line.split(",");
			Entry entry = new Entry();
			Vector<Vector<String>> words = null;
			
				if (tokens[0].equals("\"positive\""))
					entry.label = 1;
				else if (tokens[0].equals("\"negative\""))
					entry.label = -1;
				if(tokens.length>1){
					//System.out.println(tokens[1]);
					words = nlp.processText(tokens[1], true, true);}
			/*	if (tokens[0].equals("\"0\""))
				{	entry.label = 1;
				words = nlp.processText(tokens[5], true, true);}
				else if (tokens[0].equals("\"4\""))
					{entry.label = -1;
				words = nlp.processText(tokens[5], true, true);}*/
				
			
			entry.words = words;
			entry.text = line;
			InputStream is = new FileInputStream("/home/sreejat/sentimentAnalysis/models/en-sent.bin");
			SentenceModel model = new SentenceModel(is);
			SentenceDetectorME sdetector = new SentenceDetectorME(model);
			entry.sentences = sdetector.sentDetect(line);
			
			if(words!=null)
			out.add(entry);
			line = in.readLine();
			
		}
		in.close();
		System.out.println(out.size()+"======================================");
		return out;
	}
	
	/*
	 * adding feature to the "bag" of feature, with it's frequency
	 */
	void addFeature(HashMap<String, Integer> features, String feature) {
		Integer num = features.get(feature);
		if (num == null)
			features.put(feature, new Integer(1));
		else {
			features.put(feature, new Integer(num.intValue() + 1));
		}
	}
	
	/*
	 * claculates features for the dataset. 
	 * featuresSet is the setting we want to use this time.
	 */
	void calculateFeatures(Vector<Entry> entries, features feature,boolean isHybrid) throws Exception {
		
		for (int e = 0; e < entries.size(); ++e) {
			
			Entry entry = entries.get(e);
			HashMap<String, Integer> features = new HashMap<String, Integer>();
			
			switch(feature){
			case WORD:
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				
			case LEMMA:
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(3), ""));
				}
				
			
			case LEMMAPOS:
				for (int i = 0; i < entry.words.size(); ++i) {
					
					addFeature(features, cleanWord(entry.words.get(i).get(3), "__" + entry.words.get(i).get(4)));
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(3), ""));
				}
				
			//
			
			
			case LEMMAWITHPOS:
				for (int i = 0; i < entry.words.size(); ++i) {
						addFeature(features, cleanWord(entry.words.get(i).get(3), ""));
						addFeature(features, cleanWord(entry.words.get(i).get(4), ""));
				}
				
			
		
			
		
			case NEGATIONLEMMA:

				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				
				for(int i=1;i<entry.words.size();++i){
					addFeature(features,cleanWord(entry.words.get(i).get(3),"__"+entry.words.get(i-1).get(5)));
				}
		break;		
			
			
			case BIGRAMS:
for (int i = 0; i < entry.words.size(); ++i) {
					
					addFeature(features, cleanWord(entry.words.get(i).get(3), "__" + entry.words.get(i).get(4)));
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(3), ""));
				}
				for(int i=0;i<entry.words.size();++i){
					addFeature(features,cleanWord(entry.words.get(i).get(3),"__"+entry.words.get(i).get(5)));
				}
				for(int i=0;i<entry.words.size()-1;++i){
						addFeature(features, cleanWord(entry.words.get(i).get(3),entry.words.get(i+1).get(3)));
				}
				
				
				
			
			case BIGRAMPOS:
for (int i = 0; i < entry.words.size(); ++i) {
					
					addFeature(features, cleanWord(entry.words.get(i).get(3), "__" + entry.words.get(i).get(4)));
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(3), ""));
				}
				for(int i=0;i<entry.words.size();++i){
					addFeature(features,cleanWord(entry.words.get(i).get(3),"__"+entry.words.get(i).get(5)));
				}
				for(int i=0;i<entry.words.size()-1;++i){
					addFeature(features, entry.words.get(i).get(3)+"__"+entry.words.get(i).get(4)+
							"__"+entry.words.get(i+1).get(3)+"__"+entry.words.get(i+1).get(4));
				}
				
				
			
				
			
			case LENGTH:
for (int i = 0; i < entry.words.size(); ++i) {
					
					addFeature(features, cleanWord(entry.words.get(i).get(3), "__" + entry.words.get(i).get(4)));
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(3), ""));
				}
				for(int i=0;i<entry.words.size();++i){
					addFeature(features,cleanWord(entry.words.get(i).get(3),"__"+entry.words.get(i).get(5)));
				}
				//for(int i=0;i<entry.words.size()-2;++i){
					addFeature(features, ""+(entry.words.size()));
				/*
				for(int i=0;i<entry.words.size()-1;++i){
					addFeature(features, cleanWord(entry.words.get(i).get(3),entry.words.get(i+1).get(3)));
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(3), ""));
				} 	
				
			*/
		case NEGATIONBIGRAM:
			
			for (int i = 0; i < entry.words.size(); ++i) {
				addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
			
		} 
			
				if(entry.words.size()>1)
				for(int i=1;i<entry.words.size()-1;++i){
					addFeature(features,cleanWord(entry.words.get(i).get(3),"__"+entry.words.get(i-1).get(5)+entry.words.get(i+1).get(3)));
				}
				
				
			break;
			
			
			case SPECIALWORDS:

				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				
				if(entry.words.size()>1)
				for(int i=0;i<entry.words.size()-1;++i){
					//entry.words.get(i).get(0)
					addFeature(features,cleanWord("__",entry.words.get(i).get(0)));
				}
				
			break;
			case WORDTOKENS:
for (int i = 0; i < entry.words.size(); ++i) {
					
					addFeature(features, cleanWord(entry.words.get(i).get(3), "__" + entry.words.get(i).get(4)));
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				
				if(entry.words.size()>1)
				for(int i=0;i<entry.words.size();++i){
					//entry.words.get(i).get(3),
					addFeature(features,cleanWord("__",entry.words.get(i).get(2)));
				}
			
		break;	
			
			case AFFINSENTISCORE:
for (int i = 0; i < entry.words.size(); ++i) {
					
					addFeature(features, cleanWord(entry.words.get(i).get(3), "__" + entry.words.get(i).get(4)));
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(3), ""));
				}
				for(int i=0;i<entry.words.size();++i){
					addFeature(features,cleanWord(entry.words.get(i).get(3),"__"+entry.words.get(i).get(5)));
				}
				AffinSentimentUtil affin = new AffinSentimentUtil();
				for(int i=0;i<entry.words.size();i++){
					addFeature(features,"__AFFIN"+affin.extractSentiment(entry.words.get(i).get(3)));
				}
				
				
			case SWNSENTISCORE:
for (int i = 0; i < entry.words.size(); ++i) {
					
					addFeature(features, cleanWord(entry.words.get(i).get(3), "__" + entry.words.get(i).get(4)));
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(3), ""));
				}
				for(int i=0;i<entry.words.size();++i){
					addFeature(features,cleanWord(entry.words.get(i).get(3),"__"+entry.words.get(i).get(5)));
				}
				
				String phrase  = "";
				
				for(int i=0;i<entry.words.size();i++){
					addFeature(features,"__SWN"+util.extractSentiment(entry.words.get(i).get(1),entry.words.get(i).get(3)));
				}

				
								
			case LEFTSENTIMENT:

				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
			/*	for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(3), ""));
				}
				for(int i=0;i<entry.words.size();++i){
					addFeature(features,cleanWord(entry.words.get(i).get(3),"__"+entry.words.get(i).get(5)));
				}*/
				affin = new AffinSentimentUtil();
				for(int i=1;i<entry.words.size();i++){
					addFeature(features,entry.words.get(i).get(3)+"__LEFT"+affin.extractSentiment(entry.words.get(i-1).get(3)));
				}
				break;
				
			case RIGHTSENTIMENT:
/*for (int i = 0; i < entry.words.size(); ++i) {
					
					addFeature(features, cleanWord(entry.words.get(i).get(3), "__" + entry.words.get(i).get(4)));
				}*/
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
			/*	for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(3), ""));
				}
				for(int i=0;i<entry.words.size();++i){
					addFeature(features,cleanWord(entry.words.get(i).get(3),"__"+entry.words.get(i).get(5)));
				}*/
				affin = new AffinSentimentUtil();
				for(int i=0;i<entry.words.size()-1;i++){
					addFeature(features,entry.words.get(i).get(3)+"__RIGHT"+affin.extractSentiment(entry.words.get(i+1).get(3)));
				}
				break;
			/*	
			
			case NORMPHRASESENT:
for (int i = 0; i < entry.words.size(); ++i) {
					
					addFeature(features, cleanWord(entry.words.get(i).get(3), "__" + entry.words.get(i).get(4)));
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(3), ""));
				}
				for(int i=0;i<entry.words.size();++i){
					addFeature(features,cleanWord(entry.words.get(i).get(3),"__"+entry.words.get(i).get(5)));
				}
				affin = new AffinSentimentUtil();
				Map<String,String> ngrams = new HashMap<String,String>();
				for(int i=0;i<entry.words.size()-2;i++){
					phrase = entry.words.get(i).get(3)+" "+entry.words.get(i+1).get(3)+" "+entry.words.get(i+2).get(3);
					String feeling = affin.extractSentiment(phrase);
					ngrams.put(phrase,feeling);
				}
				for(int i=0;i<entry.words.size();i++){
					ArrayList<String> leftNgrams = getNgramsLeft(entry,i);
					ArrayList<String> rightNgrams = getNgramsRight(entry,i);
					String leftPosCountNorm = getPosNorm(leftNgrams,ngrams);
					String rightPosCountNorm = getPosNorm(rightNgrams,ngrams);
					String leftNegCountNorm = getNegNorm(leftNgrams,ngrams);
					String rightNegCountNorm = getNegNorm(rightNgrams,ngrams);
					addFeature(features,"_LPN"+leftPosCountNorm);
					addFeature(features,"_LNN"+leftNegCountNorm);
					addFeature(features,"_RPN"+rightPosCountNorm);
					addFeature(features,"_RNN"+rightNegCountNorm);
				}
				*/
			
			case AVGLASTPOL:
				
						for(int i=0;i<entry.words.size();i++){
						addFeature(features,entry.words.get(i).get(4)+"_"+util.extractSentimentValue(entry.words.get(i).get(3), entry.words.get(i).get(4)));
						}
						for (int i = 0; i < entry.words.size(); ++i) {
							addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
						
					} 
						break;
						
			
						
			
			case AVGPOL:
				int size = entry.sentences.length;
				double polarity = 0;
				size = entry.words.size();
				for(int i=0;i<size;i++){
					polarity+=util.extractSentimentValue(entry.words.get(i).get(3), entry.words.get(i).get(4));
				}
				polarity/=(double)size;
				addFeature(features,""+polarity);
				size = entry.sentences.length;
				polarity = 0;
				size = entry.words.size();
				for(int i=0;i<size;i++){
					polarity+=util.extractSentimentValue(entry.words.get(i).get(3), entry.words.get(i).get(4));
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				polarity/=(double)size;
				addFeature(features,""+polarity);
			break;
			case COUNTAFFINPOS:
				affin = new AffinSentimentUtil();
				double affinpos=0;
				for(int i=0;i<entry.words.size();i++){
					if(affin.extractSentiment(entry.words.get(i).get(3)).equals("positive")){
						affinpos++;
					}
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				addFeature(features,""+affinpos);
				break;
			case COUNTAFFINNEG:
				affin = new AffinSentimentUtil();
				double affinneg=0;
				for(int i=0;i<entry.words.size();i++){
					if(affin.extractSentiment(entry.words.get(i).get(3)).equals("negative")){
						affinneg++;
					}
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				addFeature(features,""+affinneg);
				break;
			case COUNTSWNPOS:
				double SWNpos = 0;
				
				for(int i=0;i<entry.words.size();i++){
					if(util.extractSentiment(entry.words.get(i).get(1),entry.words.get(i).get(3)).equals("positive")){
						SWNpos++;
					}
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				addFeature(features,""+SWNpos);
				break;
			case COUNTSWNNEG:
				double SWNneg = 0;
				
				for(int i=0;i<entry.words.size();i++){
					if(util.extractSentiment(entry.words.get(i).get(1),entry.words.get(i).get(3)).equals("negative")){
						SWNneg++;
					}
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				for(int j=0;j<SWNneg;j++)
				addFeature(features,"SWNNEG");
				break;
			case EMOTICONNUMBER:
				double emoticoncount=0;
				for(int i=0;i<entry.words.size();i++){
					if(emoticons.contains(entry.words.get(i).get(3))){
						emoticoncount++;
					}
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				for(int j=0;j<emoticoncount;j++)
				addFeature(features,"EMOTICONS");
				
			case EMOTICONPATTERN:
				for(int i=0;i<entry.words.size();i++){
					if(emoticons.contains(entry.words.get(i).get(3))){
						addFeature(features,entry.words.get(i).get(3));
					}
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				break;
			case FREQSUB:
				double freq=0;
				for(int i=0;i<entry.words.size();i++){
				if(util.extractSemantic(entry.words.get(i).get(1),entry.words.get(i).get(4)).equals("subjective")){
					freq++;
				}}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				addFeature(features, ""+freq);
				break;
			case HASHTAGWORDS:
				
				for(int i=0;i<entry.words.size();i++){
					if(entry.words.get(i).get(1).contains("#")){
						addFeature(features,entry.words.get(i).get(3));
					}
				}
			case HASHTAGCOUNT:
				if(entry.text.contains("#")){
					addFeature(features, "HASH");
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				break;
				
			case LASTWORD:
				addFeature(features, entry.words.get(entry.words.size()-1).get(3));
			case LASTWORDSHAPE:
				String lastString = entry.words.get(entry.words.size()-1).get(1);
				String phraseShape="";
				for(int i=0;i<lastString.length();i++){
					if(Character.isUpperCase(lastString.charAt(i))){
						phraseShape+="X";
					}else
					{phraseShape+="x";}
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				addFeature(features,"LAST"+"__"+phraseShape);
				break;
			case POS_COUNT:
				for(int i=0;i<entry.words.size();i++){
				addFeature(features,entry.words.get(i).get(3));
				}
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			}
				
			case SENTCOUNT:
				addFeature(features,""+entry.sentences.length);
				addFeature(features, ""+(entry.words.size()));
				break;
			case TDFFIRSTLINE:
				size = entry.sentences.length;
				polarity = 0;
				String[] firstLine = entry.sentences[0].split(" ");
				List<String> fl =  Arrays.asList(firstLine);
				int length = firstLine.length;
				
				for(int i=0;i<entry.words.size();i++){
					
					polarity+=util.extractSentimentValue(entry.words.get(i).get(3), entry.words.get(i).get(4));
				}
				polarity/=(double)size;
				addFeature(features,""+polarity);
				for (int i = 0; i < entry.words.size(); ++i) {
					addFeature(features, cleanWord(entry.words.get(i).get(1), ""));
				
			} 
				break;
				
				
			/*case TDFLASTLINE:
				size = entry.sentences.length;
				polarity = 0;
				String[] lastLine = entry.sentences[size-1].split(" ");
				length = lastLine.length;
				size = entry.words.size();
				List<String> ll = Arrays.asList(lastLine);
				if(length>0)
				for(int i=0;i<entry.words.size();i++){
					if(ll.contains(entry.words.get(size-1-i).get(3)))
					polarity+=tfCalculator(entry), entry.w)util.extractSentimentValue(entry.words.get(size-1-i).get(3), entry.words.get(size-1-i).get(4));
				}
				polarity/=(double)size;
				addFeature(features,""+polarity);*/
						}
			entry.features = features;
			
			}

		}
	 public double tfCalculator(String[] totalterms, String termToCheck) {
	        double count = 0;  //to count the overall occurrence of the term termToCheck
	        for (String s : totalterms) {
	            if (s.equalsIgnoreCase(termToCheck)) {
	                count++;
	            }
	        }
	        return count / totalterms.length;
	    }
	
	private ArrayList<String> getNgramsRight(Entry entry, int min) {
		ArrayList<String> rightNgrams = new ArrayList<String>();
		for(int i=min+1;i<entry.words.size()-2;i++){
			String phrase = entry.words.get(i).get(3)+" "+entry.words.get(i+1).get(3)+" "+entry.words.get(i+2).get(3);
			rightNgrams.add(phrase);
		}
		return rightNgrams;
	
	}

	private String getPosNorm(ArrayList<String> leftNgrams, Map<String, String> ngrams) {
		int posCount =0;
		String norm = null;
		
		for (String phrase: leftNgrams){
			
			String feeling = ngrams.get(phrase);
			if(feeling!=null)
			if(feeling.equals("positive")){
				posCount++;
			}
		}
	
		
		if(leftNgrams.size()>0){
		double posNorm = ((double)posCount)/((double)leftNgrams.size());
		if(posNorm>0.5){
			norm = "A";
		}else{
			norm = "B";
		}}else{norm = "N/A";}
		return norm;
	}
	
	private String getNegNorm(ArrayList<String> leftNgrams, Map<String, String> ngrams) {
		int negCount =0;
		String norm = null;
		for (String phrase: leftNgrams){
			
			String feeling = ngrams.get(phrase);
			if(feeling!=null)
			if(feeling.equals("negative")){
				negCount++;
			}
		}
		
		if(leftNgrams.size()>0){
			double posNorm = ((double)negCount)/((double)leftNgrams.size());
		if(posNorm>0.5){
			norm = "A";
		}else{
			norm = "B";
		}}else{norm = "N/A";}
		return norm;
	}

	private ArrayList<String> getNgramsLeft(Entry entry, int max) {
		ArrayList<String> leftNgrams = new ArrayList<String>();
		for(int i=0;i<max-2;i++){
			String phrase = entry.words.get(i).get(1)+" "+entry.words.get(i+1).get(3)+" "+entry.words.get(i+2).get(3);
			leftNgrams.add(phrase);
		}
		return leftNgrams;
	}

	/* 
	 * helper function to clean a word
	 */
	String cleanWord(String word, String suffix) {
		if (emoticons.contains(word))
			return word;
		StringBuffer newWord = new StringBuffer();
		for (int i = 0; i < word.length(); ++i) {
			if (Character.isLetter(word.charAt(i)))
				newWord.append(word.charAt(i));
		}
		if (newWord.toString().length() > 0)
			return newWord.toString() + suffix;
		return "";
	}
	
	static public boolean removeDirectory(File directory) {
		if (directory == null)
			return false;
		if (!directory.exists())
			return true;
		if (!directory.isDirectory())
			return false;

		String[] list = directory.list();

		if (list != null) {
			for (int i = 0; i < list.length; i++) {
				File entry = new File(directory, list[i]);

				if (entry.isDirectory())
				{
					if (!removeDirectory(entry))
						return false;
				}
				else
				{
					if (!entry.delete())
						return false;
				}
			}
		}
		return directory.delete();
	}
	
	class Entry {
		HashMap<String, Integer> features = new HashMap<String, Integer>();
		int label;
		String[] sentences;
		String text;
		Vector<Vector<String>> words;
		String id;
	}
	
	class CVPair{
		Vector<Entry> train = new Vector<Entry>();
		Vector<Entry> test = new Vector<Entry>();
	}
	
	class Stat{
		String classifierName;
		double accuracyTest;
		double accuracyTrain;
		
		
		void applyToAverage(Stat t, int amount) {
			accuracyTest = (accuracyTest * (amount - 1)) + t.accuracyTest;
			accuracyTest /= (double)amount;
			
			accuracyTrain = (accuracyTrain * (amount - 1)) + t.accuracyTrain;
			accuracyTrain /= (double)amount;
			if(accuracyTest<t.accuracyTest)
				accuracyTest=t.accuracyTest;
			if(accuracyTrain<t.accuracyTrain)
				accuracyTrain = t.accuracyTrain;
		}
		
		public String toString() {
			return classifierName + ": [ACC TEST: " + accuracyTest + ", ACC TRAIN: " + accuracyTrain + "]";
		}
	}
	
	public static void main(String []args) throws Exception {
		new SentimentTweet("/home/sreejat/workspace/Copy of sentiAnalysis/src/config");
	}
	
}
