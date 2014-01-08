import java.util.ArrayList;
import java.util.List;

public class Util{


        private int hits = 0;
        private List<String> words = new ArrayList<String>();
        private List<String> adjectives = new ArrayList<String>();
        private String[] tokens;


        private void addPush(String t, int score){
                hits += score;
                words.add(t);
        }

        private void multiply(String t, int score){
                hits *= score;
                adjectives.add(t);
        }

        private void init( String phrase) {
                hits = 0;
                words = new ArrayList<String>();
                adjectives = new ArrayList<String>();

                String noPunctuation = phrase.replaceAll("[^a-zA-Z ]+", " ").replaceAll(" {2,}"," ");
                tokens = noPunctuation.toLowerCase().split(" ");                
        }


        public SentimentAttributes negativity( String phrase ) {

                init(phrase);

                for(String t : tokens) {
                        if (WordList.neg5.indexOf(t) > -1) {
                                addPush(t,5);
                        } else if (WordList.neg4.indexOf(t) > -1) {
                                addPush(t,4);
                        } else if (WordList.neg3.indexOf(t) > -1) {
                                addPush(t,3);
                        } else if (WordList.neg2.indexOf(t) > -1) {
                                addPush(t,2);
                        } else if (WordList.neg1.indexOf(t) > -1) {
                                addPush(t,1);
                        }
                }

                for(String t : tokens) {
                        if (WordList.int3.indexOf(t) > -1) {
                                multiply(t, 4);
                        } else if (WordList.int2.indexOf(t) > -1) {
                                multiply(t, 3);
                        } else if (WordList.int1.indexOf(t) > -1) {
                                multiply(t, 2);
                        }
                }
                return new SentimentAttributes(hits,tokens.length);

        }



        public SentimentAttributes positivity( String phrase) {
                
                init(phrase);

                for(String t : tokens) {
                        if (WordList.pos5.indexOf(t) > -1) {
                                addPush(t,5);
                        } else if (WordList.pos4.indexOf(t) > -1) {
                                addPush(t,4);
                        } else if (WordList.pos3.indexOf(t) > -1) {
                                addPush(t,3);
                        } else if (WordList.pos2.indexOf(t) > -1) {
                                addPush(t,2);
                        } else if (WordList.pos1.indexOf(t) > -1) {
                                addPush(t,1);
                        }
                }

                for(String t : tokens) {
                        if (WordList.int3.indexOf(t) > -1) {
                                multiply(t, 4);
                        } else if (WordList.int2.indexOf(t) > -1) {
                                multiply(t, 3);
                        } else if (WordList.int1.indexOf(t) > -1) {
                                multiply(t, 2);
                        }
                }

                return new SentimentAttributes(hits,tokens.length);
        }



        public String extractSentiment( String phrase ) {
        		String feeling = null;
        		init(phrase);
                SentimentAttributes pos = positivity(phrase);
                SentimentAttributes neg = negativity(phrase);
                if(pos.length>0&&neg.length>0){
                	double resultCountPos = ((double)pos.hits/(double)pos.length);
                	double resultCountNeg = ((double)neg.hits/(double)neg.length);
                	double sentiment = resultCountPos-resultCountNeg;
                	if(sentiment>0){
                		feeling = "positive";
                	}else{
                		feeling = "negative";
                	}
                }else if(pos.length>0){
                	feeling = "positive";
                }else{
                	feeling = "negative";
                }
				return feeling;
        }
        
        
     

}
