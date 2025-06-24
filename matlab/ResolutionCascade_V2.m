% Resolution Cascade
% Author: Dominique Galvez
% Purpose: This code will create the diffraction-limited MTF for each component 
% in the optical system and allow for easy variable-switching. It will display each
% individual MTF as desired, then multiply all the MTF's together to find
% the system MTF

clear all;

fx = linspace(0,1000,1001); % "Xi" frequency in x(cycles/mm)
fy = linspace(0,1000,1001); % "Eta", frequency in y (cycles/mm)
[Xi,Eta] = meshgrid(fx); %create 2D array 
P = sqrt((Xi - 0).^2 + (Eta - 0).^2); % define rho from center of origin

promptLam = 'Wavelength [nm]:';
lambda = input(promptLam)*(10^-6);

promptDistalLens = 'Is a distal focusing lens in use? [Y/N]';
distal = input(promptDistalLens,"s");

% MTF0: Distal Focusing Lens---------------------------------------------
MTF0 = ones(1001); % default MTF of uniform 1
M1 = 1; %default magnification
if(distal=='Y')
    promptNA= 'What is the NA of the distal focusing lens?';
    NA1 = input(promptNA);
    promptMag = 'What is the (de)magnification of the distal focusing lens?';
    M1 = input(promptMag); %magnification of this optical subsystem
    
    phi = acos(lambda.*P./(2*NA1));        % define phi for MTF equation
    MTF0 = 2.*(phi-cos(phi).*sin(phi))./pi; 
    MTF0 = real(abs(MTF0));

end

% MTF 1: Hexagonal Arrayed Fiber Bundle-----------------------------------
    prompt_d_img= 'What is the fiber bundle image diameter? [mm]';
    prompt_d_core= 'What are the fiber bundle core diamters? [um]';
    prompt_d_spacing= 'What is the fiber core-to-core spacing? [um]';

    d_img = input(prompt_d_img)/M1; %magnification of this optical subsystem
    d_core = input(prompt_d_core)/1000/M1;
    d_spacing = input(prompt_d_spacing)/1000/M1;
    
    % calculated value
    % x_samp = d_spacing/2;
    % y_samp = sqrt(3)*d_spacing/2;
        theta = deg2rad(60);
        u = Xi * cos(theta) + Eta * sin(theta);
        x_samp= d_spacing;
        y_samp = sqrt(3)*d_spacing;
        u_samp = d_spacing;

        % Create core somb
        P = sqrt((Xi - 0).^2 + (Eta - 0).^2);
        sampling = abs(sinc(Xi.*x_samp).*sinc(Eta.*y_samp).*sinc(u.*u_samp));
        fiber = abs(2*besselj(1,d_core.*pi.*P)./(d_core.*pi.*P));
        MTF1 = sampling.*fiber;
        A = isnan(MTF1);
        MTF1(A) = 1; 

 

% MTF 2: Relay Optics Subsystem ---------------------------
    %USER DEFINED:
    promptNA= 'What is the NA of the relay optics?';
    NA2 = input(promptNA);
    promptMag = 'What is the magnification of the relay optics?';
    M2 = input(promptMag); %magnification of this optical subsystem

    %DO NOT TOUCH
    %P = sqrt((Xi - 0).^2 + (Eta - 0).^2); % define rho from center of origin
    phi = acos(lambda.*P./(2*NA2));        % define phi for MTF equation
    MTF2 = 2.*(phi-cos(phi).*sin(phi))./pi; 
    MTF2 = real(abs(MTF2));


%MTF 4: Detector----------------------------------------------------------
    % USER-DEFINED
    promptPixel= 'What is the pixel width of the camera detector? [um]';
    wx = (input(promptPixel)/1000)/M2; % scaled pixel size
    MTF3= abs(sinc(Xi.*wx)).*abs(sinc(Eta.*wx));

% MTF Total--------------------------------------------------------------
    MTF_final = MTF0.*MTF1.*MTF2.*MTF3;

% PLOT------------------
prompt_dim= 'Would you like plots in 2D, 3D, or both?';
dim = input(prompt_dim,"s"); 

if(strcmp('2D',dim)||strcmp('both',dim))
    figure;
    subplot(2,2,1);
        plot(fx(1:250),MTF0(1:250,1));
        title('MTF of the Focusing Optics');
        ylabel('MTF');xlabel('Frequency (cycles/mm)');
        ylim([0,1]); xlim([0,250]);
    subplot(2,2,2);
        plot(fx(1:250),MTF1(1:250,1),'r');
        hold on; plot(fy(1:250),MTF1(1,1:250),'b');
        title('MTF of the CFB'); ylabel('MTF'); xlabel('Frequency (cycles/mm)');
        legend('\eta direction','\xi direction');
        ylim([0,1]); xlim([0,250]);      
    subplot(2,2,3);
        plot(fx(1:250),MTF2(1:250,1));
        title('MTF of the Relay Optics');ylabel('MTF');xlabel('Frequency (cycles/mm)');
        ylim([0,1]); xlim([0,250]);
    subplot(2,2,4);
        plot(fx(1:250),MTF3(1:250,1));
        title('MTF of the Camera Sensor');ylabel('MTF');xlabel('Frequency (cycles/mm)');
        ylim([0,1]); xlim([0,250]);
    
    figure;
    plot(fx(1:250),real(MTF_final(1,1:250)),'r');hold on; plot(fy(1:250),MTF_final(1:250,1),'b');
    %AVG = (R + G)/2;
    xlabel('Frequency(cycles/mm)');ylabel('MTF');title('System MTF');
    XiNyquist = 1/(d_spacing); xline(XiNyquist,'--r');
    EtaNyquist = 1/(sqrt(3).*d_spacing); xline(EtaNyquist,'--b');

        relayLensNyquist = 2*NA2/(lambda); xline(relayLensNyquist,'--g');


        legend('MTF in \xi direction','MTF in \eta direction', 'CFB Nyquist in \xi','CFB Nyquist in \eta','Relay Optics Nyquist');
end
hold off;
if(strcmp('3D',dim)||strcmp('both',dim))
    figure; 
    subplot(2,2,1);
        surf(Xi(1:250,1:250),Eta(1:250,1:250),MTF0(1:250,1:250), 'EdgeColor', 'none'); colormap turbo; colorbar;
        title('MTF of the Focusing Optics');
        zlabel('MTF'); xlabel('\xi (cycles/mm)'); ylabel('\eta (cycles/mm)');
        ylim([0,250]); xlim([0,250]); zlim([0,1]);
    hold on;
    subplot(2,2,2);
        surf(Xi(1:250,1:250),Eta(1:250,1:250),MTF1(1:250,1:250), 'EdgeColor', 'none'); colormap turbo; colorbar;
        title('MTF of the CFB');  zlabel('MTF'); xlabel('\xi (cycles/mm)'); ylabel('\eta (cycles/mm)');
        ylim([0,250]); xlim([0,250]);      
    subplot(2,2,3);
        surf(Xi(1:250,1:250),Eta(1:250,1:250),MTF2(1:250,1:250), 'EdgeColor', 'none'); colormap turbo; colorbar;
        title('MTF of the Relay Optics');zlabel('MTF'); xlabel('\xi (cycles/mm)'); ylabel('\eta (cycles/mm)');
        ylim([0,250]); xlim([0,250]);
    subplot(2,2,4);
        surf(Xi(1:250,1:250),Eta(1:250,1:250),MTF3(1:250,1:250), 'EdgeColor', 'none'); colormap turbo; colorbar;
        title('MTF of the Camera Sensor');zlabel('MTF'); xlabel('\xi (cycles/mm)'); ylabel('\eta (cycles/mm)');
        ylim([0,250]); xlim([0,250]);
    hold off;

    figure;
    surf(Xi(1:250,1:250),Eta(1:250,1:250),MTF_final(1:250,1:250), 'EdgeColor', 'none'); colormap turbo; colorbar;
        title('System MTF');zlabel('MTF'); xlabel('\xi (cycles/mm)'); ylabel('\eta (cycles/mm)');
        ylim([0,250]); xlim([0,250]);
end




