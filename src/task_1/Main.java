package task_1;

import lejos.nxt.*;
import lejos.robotics.navigation.*;

public class Main {

	public static void main(String[] args) {
		test_forward(args);
//		test_turn(args);
		System.exit(0);
	}
	
	private static void test_forward(String[] args){
		//Instantiating motors .
		NXTRegulatedMotor arm_motor = new NXTRegulatedMotor(MotorPort.C);
		NXTRegulatedMotor left_motor = new NXTRegulatedMotor(MotorPort.B);
		NXTRegulatedMotor right_motor = new NXTRegulatedMotor(MotorPort.A);
		DifferentialPilot pilot = new DifferentialPilot(5.5f, 11.1f, left_motor, right_motor);  // parameters in cm
		System.out.println("Press orange\nto start.");
		boolean forward = true;
		while (Button.waitForAnyPress() == Button.ID_ENTER) {
			draw_line(arm_motor);
			double travelDistiance = 60;
			pilot.setTravelSpeed(10);  // cm per second
//			if(forward){
				pilot.travel(travelDistiance);         // cm
//				forward = false;
//			}	
//			else{
//				pilot.travel(-travelDistiance,true);
//				forward = true;
//			}
			while(pilot.isMoving())
				Thread.yield();
			System.out.println("Press orange\nto start.");
			System.out.println("Press any\nother to exit.");
			draw_line(arm_motor);
		}
		//System.exit(0);
		
	}
	
	private static void test_turn(String[] args){
		//Instantiating motors .
		NXTRegulatedMotor arm_motor = new NXTRegulatedMotor(MotorPort.C);
		NXTRegulatedMotor left_motor = new NXTRegulatedMotor(MotorPort.B);
		NXTRegulatedMotor right_motor = new NXTRegulatedMotor(MotorPort.A);
		DifferentialPilot pilot = new DifferentialPilot(5.5f, 11.1f, left_motor, right_motor);  // parameters in cm
		System.out.println("Press orange\nto start.");
		boolean forward = true;
		
		while (Button.waitForAnyPress() == Button.ID_ENTER) {
			draw_line(arm_motor);
			double travelDistiance = 20;
			pilot.setTravelSpeed(10);  // cm per second
			pilot.travelArc(80, 60);
			while(pilot.isMoving())
				Thread.yield();
			System.out.println("Press orange\nto start.");
			System.out.println("Press any\nother to exit.");
			draw_line(arm_motor);

		}
		//System.exit(0);
	}
	private static void draw_line(NXTRegulatedMotor arm_motor){		
		arm_motor.rotate(360);
		while(arm_motor.isMoving())
				Thread.yield();
	}
	
}
