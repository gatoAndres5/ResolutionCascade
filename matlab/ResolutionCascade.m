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
    MTF1=ones(1001);
    a = d_spacing*tand(30); % hexagon unit of measurement a [mm]
    r_core=d_core/2;
    MTF1 = ((r_core^2)*pi).*(besselj(1,r_core*pi.*P)./(pi*r_core.*P));
    MTF1 = abs(MTF1);
        max_value = max(MTF1(:));
        if max_value ~= 1
            MTF1 = MTF1 ./ max_value;
        end
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
    subplot(1,4,1);
        plot(fx(1:250),MTF0(1:250,1));
        title('MTF of the Focusing Optics');
        ylabel('MTF');xlabel('Frequency (cycles/mm)');
        ylim([0,1]); xlim([0,250]);
    subplot(1,4,2);
        plot(fx(1:250),MTF1(1:250,1));
        title('MTF of the CFB'); ylabel('MTF'); xlabel('Frequency (cycles/mm)');
        ylim([0,1]); xlim([0,250]);      
    subplot(1,4,3);
        plot(fx(1:250),MTF2(1:250,1));
        title('MTF of the Relay Optics');ylabel('MTF');xlabel('Frequency (cycles/mm)');
        ylim([0,1]); xlim([0,250]);
    subplot(1,4,4);
        plot(fx(1:250),MTF3(1:250,1));
        title('MTF of the Camera Sensor');ylabel('MTF');xlabel('Frequency (cycles/mm)');
        ylim([0,1]); xlim([0,250]);
    
    figure;
    plot(fx(1:250),real(MTF_final(1:250,2)));xlabel('Frequency(cycles/mm)');ylabel('MTF');title('System MTF');
    hold on;
 
        if(distal=='Y')
        distalLensNyquist = 2*NA1/(lambda); xline(distalLensNyquist,'--r');
        end
        coreNyquist = 1/(d_spacing*2); xline(coreNyquist,'--b');
        relayLensNyquist = 2*NA2/(lambda); xline(relayLensNyquist,'--g');
    
        if(distal=='Y')
        legend('Theoretical MTF','Distal Lens Nyquist','CFB Nyquist','Relay Optics Nyquist');
        else
        legend('Theoretical MTF','CFB Nyquist','Relay Optics Nyquist');
        end
end
hold off;
if(strcmp('3D',dim)||strcmp('both',dim))
    figure; 
    subplot(1,4,1);
        surf(Xi(1:250,1:250),Eta(1:250,1:250),MTF0(1:250,1:250), 'EdgeColor', 'none'); colormap turbo; colorbar;
        title('MTF of the Focusing Optics');
        zlabel('MTF'); xlabel('\xi (cycles/mm)'); ylabel('\eta (cycles/mm)');
        ylim([0,250]); xlim([0,250]); zlim([0,1]);
    hold on;
    subplot(1,4,2);
        surf(Xi(1:250,1:250),Eta(1:250,1:250),MTF1(1:250,1:250), 'EdgeColor', 'none'); colormap turbo; colorbar;
        title('MTF of the CFB');  zlabel('MTF'); xlabel('\xi (cycles/mm)'); ylabel('\eta (cycles/mm)');
        ylim([0,250]); xlim([0,250]);      
    subplot(1,4,3);
        surf(Xi(1:250,1:250),Eta(1:250,1:250),MTF2(1:250,1:250), 'EdgeColor', 'none'); colormap turbo; colorbar;
        title('MTF of the Relay Optics');zlabel('MTF'); xlabel('\xi (cycles/mm)'); ylabel('\eta (cycles/mm)');
        ylim([0,250]); xlim([0,250]);
    subplot(1,4,4);
        surf(Xi(1:250,1:250),Eta(1:250,1:250),MTF3(1:250,1:250), 'EdgeColor', 'none'); colormap turbo; colorbar;
        title('MTF of the Camera Sensor');zlabel('MTF'); xlabel('\xi (cycles/mm)'); ylabel('\eta (cycles/mm)');
        ylim([0,250]); xlim([0,250]);
    hold off;

    figure;
    surf(Xi(1:250,1:250),Eta(1:250,1:250),MTF_final(1:250,1:250), 'EdgeColor', 'none'); colormap turbo; colorbar;
        title('System MTF');zlabel('MTF'); xlabel('\xi (cycles/mm)'); ylabel('\eta (cycles/mm)');
        ylim([0,250]); xlim([0,250]);
end




