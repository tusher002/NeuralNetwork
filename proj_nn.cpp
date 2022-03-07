/*
 * Sample program for CDA5125
 *
 * A generic 3 level feedforward neural network from scratch
 * 
 * Change N0 (input size), N1 (size of the hidden layer),
 *        and N2 (size of the output layer) to change the neural network
 * 
 * driver program train 
 */
#include<sys/time.h>
#include<time.h>
#include <iostream>
#include <math.h> 
#include <stdlib.h>
#include <bits/stdc++.h>
#include <random>
#include <chrono>
#include <ctime>
#include <xmmintrin.h>
using namespace std;

#define A  1.7159
#define B  0.6666
#define N0  784
#define N1  1000
#define N2  500
#define N3  4

ofstream MyFile;
int counts[N3];

double IN[N0];
double W0[N0][N1];
double B1[N1];
double HS[N1];
double HO[N1];
double W1[N1][N2];
double B2[N2];
double HS2[N1];
double HO2[N1];
double W2[N2][N3];
double B3[N3];
double OS[N2];
double OO[N2];

double err;
double rate = 0.0001;


int n_rows=28;
int n_cols=28;
int  image[60000][28][28];
int  label[60000];
int  predicted_label[20679];

int selected_labels_index[20679];
double selected_Y_label[20679][4];
double selected_X_data[20679][784];

int cnt_Z=0;

void get_selected_labels(){
    for(int i=0,k=0;i<50000;i++){
        if(label[i]<4){
            counts[label[i]] += 1;
            selected_labels_index[k]=i;
            k++;
        }
    }
}

// 20679
void get_selected_X_data(){
    for(int i=0;i<20679;i++){
        for (int j=0; j<28;j++){
            for(int k=0;k<28;k++){
                int real_index=selected_labels_index[i];
                selected_X_data[i][k+j*28]=(image[real_index][j][k]/127.5)-1;
            }
        }  
    }
}

// 20679
void get_selected_Y_label(){
    for(int i=0;i<20679;i++){
        int real_index=selected_labels_index[i];
        for (int j =0; j<4; j++){
            if (j==label[real_index]){
                selected_Y_label[i][j]=A;
            }
            else{
                selected_Y_label[i][j]=-A;
            }
        } 
    }
}


int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void display_image_by_id(int image_id){  
    for(int r=0;r<n_rows;++r)
    {
        for(int c=0;c<n_cols;++c)
        {
            if (image[image_id][r][c] ==0) {
                cout<<".";
            }
            else{
                cout<<"@";
            }
            //cout<<image[i][r][c]<<" ";
        }
        cout<<endl;
    }
    cout<<"-----------------------------";
    cout<<endl;
    cout<<endl;
    cout<<endl;            
}

void display_label_by_id(int label_id){
    cout<<label[label_id]<<endl;
}
void training_image(/*string full_path*/){
    ifstream file ("train-images.idx3-ubyte");
    if (file.is_open()){
        int magic_number=0;
        int number_of_images=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    image[i][r][c] = temp;
                }
            }

        }  
        //display_image_by_id(7000);
    }
    else{
        cout<<"Unable to openfile \n";
        exit(0);
    }
}

void training_label(){
    int number_of_images=0;
    ifstream file ("train-labels.idx1-ubyte");
    if (file.is_open())
    {
        int magic_number=0;        
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        for(int i=0;i<number_of_images;++i)
        {
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            label[i]= temp; 
        }  
        //display_label_by_id(7000);
    }
}

double sigmoid(double x)
{
    //cout<<x<<endl;
    double xx = A*tanh(B*x);
	return xx;
}


// forward progagation with input: input[N0]
void forward(double *input)
{

        for (int i = 0; i<N0; i++) 
		IN[i] = input[i];

        // compute the weighted sum HS in the hidden layer
        for (int i=0; i<N1; i++) {
		HS[i] = B1[i];
        
	}
        for (int i=0; i<N1; i++) {
		for (int j=0; j<N0; j++){
			HS[i] += IN[j]*W0[j][i];
            //cout<<W0[j][i]<<endl;
            }
	}

        // Comput the output of the hidden layer, HO[N1];

        for (int i=0; i<N1; i++) {
		HO[i] = sigmoid(HS[i]);
        //cout<<HS[i]<<"   "<<HO[i]<<endl;
	}







        // compute the weighted sum HS in the hidden layer
        for (int i=0; i<N2; i++) {
		HS2[i] = B2[i];
        
	}
        for (int i=0; i<N2; i++) {
		for (int j=0; j<N1; j++){
			HS2[i] += HO[j]*W1[j][i];
            //cout<<W0[j][i]<<endl;
            }
	}

        // Comput the output of the hidden layer, HO[N1];

        for (int i=0; i<N1; i++) {
		HO2[i] = sigmoid(HS2[i]);
        //cout<<HS[i]<<"   "<<HO[i]<<endl;
	}


        // compute the weighted sum OS in the output layer
        for (int i=0; i<N3; i++) {
		OS[i] = B3[i];
	}
        for (int i=0; i<N3; i++) {
		for (int j=0; j<N2; j++)
			OS[i] += HO2[j]*W2[j][i];
	}

        // Comput the output of the output layer, OO[N2];

        for (int i=0; i<N3; i++) {
		OO[i] = sigmoid(OS[i]);
        
	}
}

void print_val(int i){
    forward(&(selected_X_data[i][0]));
    for (int i=0;i<N3;i++){
        string str;
        str = "OO[" + to_string(i) + "] = " + to_string(OO[i]) + "\n";
        MyFile << str;
    }
    string str;
    time_t my_time = time(NULL);
    str = ctime(&my_time);
    MyFile << str;
}


double dE_OO[N3];
double dOO_OS[N3];
double dE_OS[N3];
double dE_B3[N3];
double dE_W2[N2][N3];

double dE_HO[N1];
double dHO_HS[N1];
double dE_HS[N1];
double dE_B1[N1];
double dE_W0[N0][N1];

double dE_HO2[N2];
double dHO2_HS2[N2];
double dE_HS2[N2];
double dE_B2[N2];
double dE_W1[N1][N2];



// 
double backward(double *O, double *Y)
{
        // compute error
	err = 0.0;
        for (int i=0; i<N3; i++) 
		err += (O[i] - Y[i])*(O[i]-Y[i]);
	err = err / N3;

        // compute dE_OO
        for (int i=0; i<N3; i++) 
		dE_OO[i] = (O[i] - Y[i])*2.0/N3;

        // compute dOO_OS = OO dot (1-OO)
        for (int i=0; i<N3; i++)
		dOO_OS[i] = A*B*(1- ((OO[i]/A) * (OO[i]/A)));

        // compute dE_OS = dE_OO dot dOO_OS
        for (int i=0; i<N3; i++)
		dE_OS[i] = dE_OO[i] * dOO_OS[i];

        // compute dE_B2 = dE_OS
        for (int i=0; i<N3; i++)
		dE_B3[i] = dE_OS[i];

        // compute dE_W1
        for (int i=0; i<N2; i++)
		for (int j = 0; j<N3; j++) 
			dE_W2[i][j] = dE_OS[j]*HO2[i];

	// compute dE_HO
	for (int i=0; i<N2; i++) {
		dE_HO2[i] = 0;
		for (int j = 0; j<N3; j++)
			dE_HO2[i] += dE_OS[j]*W2[i][j];
	}

        // compute dHO_HS = HO dot (1-HO)
        for (int i=0; i<N2; i++)
		dHO2_HS2[i] = A*B*(1- ((HO2[i]/A) * (HO2[i]/A)));

        // compute dE_HS = dE_HO dot dHO_HS
        for (int i=0; i<N2; i++)
		dE_HS2[i] = dE_HO2[i] * dHO2_HS2[i];

        // compute dE_B1 = dE_HS
        for (int i=0; i<N2; i++)
		dE_B2[i] = dE_HS2[i];

        // compute dE_W0
        for (int i=0; i<N1; i++)
		for (int j = 0; j<N2; j++) 
			dE_W1[i][j] = dE_HS2[j]*HO[i];
	
    // compute dE_HO
	for (int i=0; i<N1; i++) {
		dE_HO[i] = 0;
		for (int j = 0; j<N2; j++)
			dE_HO[i] += dE_HS2[j]*W1[i][j];
	}

        // compute dHO_HS = HO dot (1-HO)
        for (int i=0; i<N1; i++)
		dHO_HS[i] = A*B*(1- ((HO[i]/A) * (HO[i]/A)));

        // compute dE_HS = dE_HO dot dHO_HS
        for (int i=0; i<N1; i++)
		dE_HS[i] = dE_HO[i] * dHO_HS[i];

        // compute dE_B1 = dE_HS
        for (int i=0; i<N1; i++)
		dE_B1[i] = dE_HS[i];

        // compute dE_W0
        for (int i=0; i<N0; i++)
		for (int j = 0; j<N1; j++) 
			dE_W0[i][j] = dE_HS[j]*IN[i];











	for (int i=0; i<N0; i++)
		for (int j=0; j<N1; j++)
			W0[i][j] = W0[i][j] - rate * dE_W0[i][j];

	for (int i=0; i<N1; i++)
		B1[i] = B1[i] - rate * dE_B1[i];

	for (int i=0; i<N1; i++)
		for (int j=0; j<N2; j++)
			W1[i][j] = W1[i][j] - rate * dE_W1[i][j];

	for (int i=0; i<N2; i++)
		B2[i] = B2[i] - rate * dE_B2[i];

    for (int i=0; i<N2; i++)
		for (int j=0; j<N3; j++)
			W2[i][j] = W2[i][j] - rate * dE_W2[i][j];

	for (int i=0; i<N3; i++)
		B3[i] = B3[i] - rate * dE_B3[i];

}  
double forward_time[100];
double backward_time[100];
double forward_time_avg;
double backward_time_avg;

void duration_f(struct timespec *b, struct timespec *c, int i)
{
	long long r = c->tv_nsec - b->tv_nsec;
        r += ((long long)(c->tv_sec - b->tv_sec) ) * 1000000000;
	forward_time[i] = r;
}

void duration_b(struct timespec *b, struct timespec *c, int i)
{
	long long r = c->tv_nsec - b->tv_nsec;
        r += ((long long)(c->tv_sec - b->tv_sec) ) * 1000000000;
	backward_time[i] = r;
}


void train(int iter)
{
	for (int i = 0; i< iter; i++) {
		//int ii = random () % 4;
		int ii = i % 20679;
                //int ii= 3;
        struct timespec b, e;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b);
		forward(&(selected_X_data[ii][0]));
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &e);  
        duration_f(&b, &e, i);

        int max_ho_ind=-1;
        int flag=1;
        for (int j=0; j<N3; j++){
            if(OO[j]>0){
                max_ho_ind=j;
                flag=0;
            }
        }
        for (int j=0; j<N3; j++){
            if(OO[j]>0 && flag==0 && j!=max_ho_ind){ 
                max_ho_ind=-1;
                flag=1;
            }
        }

        if (max_ho_ind!=-1)
            predicted_label[ii] = max_ho_ind;        
        else
            predicted_label[ii] = -1;

        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &b);
		backward(OO, &(selected_Y_label[ii][0]));
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &e);  
        duration_b(&b, &e, i);

		if (i % 10000 == 0 && i!=0){
            cout << "Iter " << i << ": err =" << err << "\n"; 
            int rand_ind=rand()%20679;
            MyFile.open("output_hudai.txt", std::ios_base::app); // append instead of overwrite
            string str;
            int real_index=selected_labels_index[rand_ind];
            str = "Iter "+ to_string(i)+": err = "+to_string(err)+", Y= "+to_string(label[real_index])+"\n";
            MyFile << str;
            print_val(rand_ind);
            MyFile.close();
        }
        if (i % 20679 == 0 && i!=0){
            // if (i % 20000 == 0){
                int match = 0;
                float accuracy;
                for(int j = 0; j<20679; j++){
                    int real_index=selected_labels_index[j];
                    if(label[real_index] == predicted_label[j])
                    match++;
                }
                accuracy = match*100.0/20679;                
                cout << "Training accuracy: " << accuracy<<"%" <<"\n";
        }
	}
}

int main(int argc, char *argv[]) 
{
    for(int i=0; i<N3; i++){
        counts[i] = 0;
    }

    training_image();
    training_label();

    get_selected_labels();
    get_selected_X_data();
    get_selected_Y_label();

    MyFile.open("output_hudai.txt");
    for(int i=0; i<N3; i++)
    {   
        string str;
        str = "c["+ to_string(i)+"] = "+to_string(counts[i])+"\n";
        MyFile << str;

    }
    MyFile.close();
    
 
	// randomize weights
    random_device rd;  // Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    uniform_real_distribution<> dis(-0.05, 0.05);
    uniform_real_distribution<> dis_b(-0.05, 0.05);
    for (int j = 0; j<N1; j++)        
    {
        for (int i = 0; i<N0; i++)
        {
            W0[i][j] = dis(gen);
        }
    }
    for (int i = 0; i<N2; i++){
        B2[i] = dis_b(gen);
    }

    for (int j = 0; j<N2; j++)        
    {
        for (int i = 0; i<N1; i++)
        {
            W1[i][j] = dis(gen);
        }
    }

    for (int i = 0; i<N3; i++){
        B3[i] = dis_b(gen);
    }
    
    for (int j = 0; j<N3; j++)
    {
        for (int i = 0; i<N2; i++)
        {
            W2[i][j] = dis(gen);
        }
    }
			
    time_t my_time = time(NULL);
    cout<<"Training begins: "<<ctime(&my_time)<<endl;
	if (argc == 2) train(atoi(argv[1]));
        else train(100);
    
    forward_time_avg = 0;
    backward_time_avg = 0;

    for(int i=0; i<100; i++)
    {
        forward_time_avg+= forward_time[i];
        backward_time_avg+= backward_time[i];
    }
    
    forward_time_avg /= 100;
    backward_time_avg /= 100;
    cout<<"Forward avg: "<<forward_time_avg<<endl;
    cout<<"Backward avg: "<<backward_time_avg<<endl;
    return 0;
}