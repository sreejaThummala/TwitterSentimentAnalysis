import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.Vector;

import net.didion.jwnl.JWNL;
import net.didion.jwnl.data.IndexWord;
import net.didion.jwnl.data.POS;
import net.didion.jwnl.dictionary.Dictionary;
import opennlp.tools.chunker.ChunkerME;
import opennlp.tools.chunker.ChunkerModel;
import opennlp.tools.namefind.NameFinderME;
import opennlp.tools.namefind.TokenNameFinderModel;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.Span;
import weka.core.Stopwords;

/*
 * pre processing
 */

public class NLP {
	
	TokenizerME tokenizer;
	ChunkerME chunker;
	POSTaggerME pos;
	NameFinderME orgTagger;
	NameFinderME perTagger;
	NameFinderME locTagger;
	
	String[] nNegators = new String[]{
			"Never",
			"Neither",
			"Nobody",
			"No",
			"None",
			"Nor",
			"Nothing",
			"Nowhere"};

	public NLP(String openNlpModelsPath, String wordnetConfigFilePath) throws Exception{
		
		URL tokModel = new URL(openNlpModelsPath + "/en-token.bin");
		URL chunkerModel = new URL(openNlpModelsPath + "/en-chunker.bin");
		URL posModel = new URL(openNlpModelsPath + "/en-pos-maxent.bin");
		URL orgModel = new URL(openNlpModelsPath + "/en-ner-organization.bin");
		URL perModel = new URL(openNlpModelsPath + "/en-ner-person.bin");
		
		URL locModel = new URL(openNlpModelsPath + "/en-ner-location.bin");
		JWNL.initialize(new FileInputStream(wordnetConfigFilePath));
		
		tokenizer = new TokenizerME(getTokenizerModel(tokModel));
		chunker = new ChunkerME(getChunkerModel(chunkerModel));
		pos = new POSTaggerME(getPOSModel(posModel));
		
		orgTagger = new NameFinderME(getNameFinderModel(orgModel));
		perTagger = new NameFinderME(getNameFinderModel(perModel));
		locTagger = new NameFinderME(getNameFinderModel(locModel));
		
	}
	
	public Vector<Vector<String>> processText(String text, boolean runNer, boolean runChunker) throws Exception{
		Vector<Vector<String>> output = new Vector<Vector<String>>();
	
		String []tokens = text.split(" ");
		//Remove stopwords
		Stopwords stopwords = new Stopwords(); 
		
		
		Vector<String> toks = new Vector<String>();
		for (int i = 0; i < tokens.length; ++i) {
			Vector<String> tokVec = new Vector<String>();
			if(!Stopwords.isStopword(tokens[i]))	
			if (tokens[i].trim().length() > 0) {
				
				if(tokens[i].contains("?")){
					tokens[i]=tokens[i].replace("?","");
					if(tokens[i].startsWith("\"")&&tokens[i].endsWith("\"")){
						tokVec.add("SPECIAL");
						tokens[i]=tokens[i].replace("\"","");
					}else if(tokens[i].startsWith("_")&&tokens[i].endsWith("_")){
						tokVec.add("SPECIAL");
						tokens[i]=tokens[i].replace("_","");
					}else if(tokens[i].startsWith("*")&&tokens[i].endsWith("*")){
						tokVec.add("SPECIAL");
						tokens[i]=tokens[i].replace("*", "");
					}else {
						tokVec.add("NOTSPECIAL");
					}
					
					tokVec.add(tokens[i].trim());
					tokens[i]= clean(tokens[i]);
					
					tokVec.add(tokens[i].trim());
					tokVec.add("QUESTION");
					
				}else if(tokens[i].contains("!")){
					tokens[i]=tokens[i].replace("!", "");
					if(tokens[i].startsWith("\"")&&tokens[i].endsWith("\"")){
						tokVec.add("SPECIAL");
						tokens[i]=tokens[i].replace("\"","");
					}else if(tokens[i].startsWith("_")&&tokens[i].endsWith("_")){
						tokVec.add("SPECIAL");
						tokens[i]=tokens[i].replace("_","");
					}else if(tokens[i].startsWith("*")&&tokens[i].endsWith("*")){
						tokVec.add("SPECIAL");
						tokens[i]=tokens[i].replace("*", "");
					}else{
						tokVec.add("NOTSPECIAL");
					}
					tokVec.add(tokens[i].trim());
					tokens[i]= clean(tokens[i]);
					tokVec.add(tokens[i].trim());
					tokVec.add("EXCLAIM");
				}else {
					tokVec.add("NOTSPECIAL");
					tokVec.add(tokens[i].trim());
					tokens[i]= clean(tokens[i]);
					tokVec.add(tokens[i].trim());
					tokVec.add("NO_PUNCT");
				}
				output.add(tokVec);
				toks.add(tokens[i].trim());
			}
		}
		
		//System.out.println(output.size());
		String []words = new String[output.size()];
		for(int i = 0 ; i < output.size() ; ++i){
			words[i] = output.get(i).get(0);
		}
		
		String []posTags = pos.tag(toks.toArray(new String[] { }));
		for(int i = 0 ; i < posTags.length ; ++i){
			POS pos = null;
			if(posTags[i].startsWith("VB")){
				pos = POS.VERB;
			}else if(posTags[i].startsWith("NN")){
				pos = POS.NOUN;
			}else if(posTags[i].startsWith("JJ")){
				pos = POS.ADJECTIVE;
			}else if(posTags[i].startsWith("RB")){
				pos = POS.ADVERB;
			}
			IndexWord indexWord = null;
			if(pos != null){
				/*not a  URL and a mashed up word*/
				//System.out.println("POS notnull");
				if (toks.get(i).indexOf("://") == -1 && toks.get(i).length() < 20)
					indexWord = Dictionary.getInstance().lookupIndexWord(pos, toks.get(i));
			} 
			if(indexWord != null)
				output.get(i).add(indexWord.getLemma());
			else
				output.get(i).add(toks.get(i).toLowerCase());
			output.get(i).add(posTags[i]);
		}
		for(int i=0;i<output.size()-1;i++){
			String negationTester = output.get(i).get(0);
			if(Arrays.asList(nNegators).contains(negationTester)){
				output.get(i+1).add("NEG");
			} else {
				output.get(i+1).add("POS");
			}
			
		} 
		output.get(0).add("POS");
		if (runChunker) {
			String []chunkTags = chunker.chunk(words, posTags);
			for(int i = 0 ; i < chunkTags.length ; ++i){
				output.get(i).add(chunkTags[i]);
			}
		}
		if (runNer) {
			Span orgSpans[] = orgTagger.find(words);
			Span perSpans[] = perTagger.find(words);
			Span locSpans[] = locTagger.find(words);
			orgTagger.clearAdaptiveData();
			perTagger.clearAdaptiveData();
			locTagger.clearAdaptiveData();
			
			String []ner = new String[output.size()];
			for(int i = 0 ; i < ner.length ; ++i)
				ner[i] = "O";
			for(int i = 0 ; i < orgSpans.length ; ++i){
				boolean first = true;
				for(int j = orgSpans[i].getStart() ; j < orgSpans[i].getEnd() ; ++j){
					if(first)
						ner[j] = "B-ORG";
					else
						ner[j] = "I-ORG";
					first = false;
				}
			}
			for(int i = 0 ; i < perSpans.length ; ++i){
				boolean first = true;
				for(int j = perSpans[i].getStart() ; j < perSpans[i].getEnd() ; ++j){
					if(first)
						ner[j] = "B-PER";
					else
						ner[j] = "I-PER";
					first = false;
				}
			}
			for(int i = 0 ; i < locSpans.length ; ++i){
				boolean first = true;
				for(int j = locSpans[i].getStart() ; j < locSpans[i].getEnd() ; ++j){
					if(first)
						ner[j] = "B-LOC";
					else
						ner[j] = "I-LOC";
					first = false;
				}
			}
			for(int i = 0 ; i < ner.length ; ++i){
				output.get(i).add(ner[i]);
			}
		}
		
		return output;
	}

	private String clean(String token) {
		String cleanedToken = null;
		if(token.contains("@")){
			cleanedToken = "USER";
		}else cleanedToken = token;
		cleanedToken.replace(";","");
		cleanedToken.replace(".","");
		cleanedToken.replace(":","");
		
		String pattern = "(.)(?=\\1{2})";
        cleanedToken.replaceAll(pattern, "");
        
		return cleanedToken;
	}

	public static TokenizerModel getTokenizerModel(URL name) {
		try {
			return new TokenizerModel(new DataInputStream(name.openStream()));
		} catch (IOException E) {
			E.printStackTrace();
			throw new RuntimeException("OpenNLP Tokenizer can not be initialized!", E);
		}
	}
	
	public static ChunkerModel getChunkerModel(URL name) {
		try {
			return new ChunkerModel(new DataInputStream(name.openStream()));
		} catch (IOException E) {
			E.printStackTrace();
			throw new RuntimeException("OpenNLP Tokenizer can not be initialized!", E);
		}
	}
	
	public static POSModel getPOSModel(URL name) {
		try {
			return new POSModel(new DataInputStream(name.openStream()));
		} catch (IOException E) {
			E.printStackTrace();
			throw new RuntimeException("OpenNLP Tokenizer can not be initialized!", E);
		}
	}
	
	public static TokenNameFinderModel getNameFinderModel(URL name) {
		try {
			return new TokenNameFinderModel(new DataInputStream(name.openStream()));
		} catch (IOException E) {
			E.printStackTrace();
			throw new RuntimeException("OpenNLP Tokenizer can not be initialized!", E);
		}
	}
	
	
	static class Document{
		Vector<String> sentences = new Vector<String>();
		String docId;
		String headline;
		String dateLine;
	}
}
