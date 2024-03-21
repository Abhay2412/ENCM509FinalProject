% speaker_reco_demo
% Created by A. Alexander, EPFL
% Modified by J. Richiardi
% Modifie by S. Yanushkevich

%define the number of Gaussian invariants - could be modified
No_of_Gaussians=10;
%Reading in the data 
%Use wavread from matlab 
disp('-------------------------------------------------------------------');
disp('                    Speaker recognition Demo');
disp('                    using GMM');
disp('-------------------------------------------------------------------');

%-----------reading in the training data----------------------------------
[training_data1,Fs1]=audioread('./Imposter_AI_Data/Trudeau_Imposter/1.wav');
[training_data2,Fs2]=audioread('./Imposter_AI_Data/Trudeau_Imposter/2.wav');
[training_data3,Fs3]=audioread('./Imposter_AI_Data/Trudeau_Imposter/3.wav');

%------------reading in the test data-----------------------------------
[testing_data1,Fs4]=audioread('./Imposter_AI_Data/Trump_Imposter/1.wav');
[testing_data2,Fs5]=audioread('./Imposter_AI_Data/Trump_Imposter/2.wav');
[testing_data3,Fs6]=audioread('./Imposter_AI_Data/Trump_Imposter/3.wav');

disp('Completed reading taining and testing data (Press any key to continue)');
pause;

%Fs=8000;   %uncoment if you cannot obtain the feature number from wavread above

%-------------feature extraction------------------------------------------
training_features1=melcepst(training_data1,Fs1);
training_features2=melcepst(training_data2,Fs2);
training_features3=melcepst(training_data3,Fs3);

disp('Completed feature extraction for the training data (Press any key to continue)');
pause;


testing_features1=melcepst(testing_data1,Fs4);
testing_features2=melcepst(testing_data2,Fs5);
testing_features3=melcepst(testing_data3,Fs6);

disp('Completed feature extraction for the testing data (Press any key to continue)');
pause;

%-------------training the input data using GMM-------------------------
%training input data, and creating the models required
disp('Training models with the input data (Press any key to continue)');

[mu_train1,sigma_train1,c_train1]=gmm_estimate(training_features1',No_of_Gaussians);
disp('Completed Training Speaker 1 model (Press any key to continue)');
pause;

[mu_train2,sigma_train2,c_train2]=gmm_estimate(training_features2',No_of_Gaussians);
disp('Completed Training Speaker 2 model (Press any key to continue)');
pause;

[mu_train3,sigma_train3,c_train3]=gmm_estimate(training_features3',No_of_Gaussians);
disp('Completed Training Speaker 3 model (Press any key to continue)');
pause;


disp('Completed Training ALL Models  (Press any key to continue)');

pause;
%-------------------------testing against the input data-------------- 

%testing against the first model
[lYM,lY]=lmultigauss(testing_features1', mu_train1,sigma_train1,c_train1);
A(1,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features2', mu_train1,sigma_train1,c_train1);
A(1,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features3', mu_train1,sigma_train1,c_train1);
A(1,3)=mean(lY);

%testing against the second model
[lYM,lY]=lmultigauss(testing_features1', mu_train2,sigma_train2,c_train2);
A(2,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features2', mu_train2,sigma_train2,c_train2);
A(2,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features3', mu_train2,sigma_train2,c_train2);
A(2,3)=mean(lY);

%testing against the third model
[lYM,lY]=lmultigauss(testing_features1', mu_train3,sigma_train3,c_train3);
A(3,1)=mean(lY);
[lYM,lY]=lmultigauss(testing_features2', mu_train3,sigma_train3,c_train3);
A(3,2)=mean(lY);
[lYM,lY]=lmultigauss(testing_features3', mu_train3,sigma_train3,c_train3);
A(3,3)=mean(lY);

disp('Results in the form of confusion matrix for comparison');
disp('Each column i represents the test recording of Speaker i');
disp('Each row i represents the training recording of Speaker i');
disp('The diagonal elements corresponding to the same speaker');
disp('-------------------------------------------------------------------');
A
disp('-------------------------------------------------------------------');
% confusion matrix in color
figure; imagesc(A); colorbar;