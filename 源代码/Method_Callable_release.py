import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from matplotlib.figure import Figure
import copy

class Method_pf_tsbs_tpebs3:
    @staticmethod
    def polar_power_flow(Sbase, Ubase, f, BusData, LineData):
        Zbase = (Ubase ** 2) / Sbase

        # Transforms transient reactance X' into transient admittance Y'

        for i in range(len(BusData['TransientSusceptance'])):
            if BusData['TransientSusceptance'][i] != 0:
                BusData['TransientSusceptance'][i] = 1 / BusData['TransientSusceptance'][i]

        # Save Line output
        # Create dictionary from line data list
        LineData_1 = {
            'from': None, 'to': None, 'R': None, 'Xl': None, 'Xc': None,
            'tap': None, 'tapmin': None, 'tapmax': None, 'phi': None,
            'Pkm': None, 'Pmk': None, 'Qkm': None, 'Qmk': None,
            'ActiveLoss_km': None, 'ReactiveLoss_km': None
        }

        for key in LineData.keys():
            temp = LineData[key]
            LineData_1[key] = temp[:]
            del temp

        # Adjust bus numbers for Python indexing (start from 0)
        for k in range(len(LineData['from'])):
            LineData['from'][k] -= 1
        for k in range(len(LineData['to'])):
            LineData['to'][k] -= 1

        # ---------------------------------------------------------------------
        # Extract system size
        nb = len(BusData['num'])  # Number of buses
        nl = len(LineData['from'])  # Number of lines
        npv = 0
        npq = 0
        for i in range(nb):
            if BusData['type'][i] == 1:
                npv += 1
            elif BusData['type'][i] == 0:
                npq += 1

        # ---------------------------------------------------------------------
        # Normalize and compute net power for each bus

        temp3 = []
        temp4 = []
        bus = {'P': None, 'Q': None, 'bsh': None}
        for i in range(nb):
            temp3.append((BusData['Pgen'][i] - BusData['Pload'][i]) / Sbase)  # calculate pu value
            temp4.append((BusData['Qgen'][i] - BusData['Qload'][i]) / Sbase)  # calculate pu value
            BusData['Pload'][i] = BusData['Pload'][i] / Sbase  # calculate pu value
            BusData['Qload'][i] = BusData['Qload'][i] / Sbase  # calculate pu value
            bus['P'] = temp3[:]
            bus['Q'] = temp4[:]
        del temp3
        del temp4

        temp5 = []
        for i in range(nb):
            BusData['voltage'][i] = BusData['voltage'][i] / Ubase  # calculate voltage pu value
            BusData['angle'][i] = BusData['angle'][i] * (np.pi / 180)  # Convert degrees to radians
            BusData['Qshunt'][i] = BusData['Qshunt'][i] / Sbase  # calculate Q pu value
            temp5.append(BusData['Qshunt'][i])
            bus['bsh'] = temp5[:]
        del temp5

        # those two lines can save BusData
        # df = pd.DataFrame(BusData)
        # df.to_excel('BusData.xlsx', index=False)

        # ---------------------------------------------------------------------
        # Convert line parameters to per-unit (pu) system

        for i in range(nl):  # Transformer taps 1 means voltage unchanged
            if LineData['tap'][i] == 0:
                LineData['tap'][i] = 1

        num_tap_normal = 0
        num_phase_shift = 0
        temp3 = []
        for i in range(nl):
            LineData['R'][i] = LineData['R'][i] / Zbase  # ohm to pu
            LineData['Xl'][i] = (LineData['Xl'][i] * 2 * np.pi * f) / Zbase
            LineData['Xc'][i] = (LineData['Xc'][i] * 2 * np.pi * f * Zbase) / 2
            temp3.append(LineData['Xc'][i])  # Calculate Lsh
            LineData['phi'][i] = LineData['phi'][i] * (np.pi / 180)  # Phase-shifter angle to radians
            if LineData['tap'][i] != 1.0:
                num_tap_normal += 1
            elif LineData['phi'][i] != 0.0:
                num_phase_shift += 1

        # ---------------------------------------------------------------------
        # Build Y, G, B matrices for power flow analysis

        Y = np.zeros((nb, nb), dtype=complex)
        branch = {'g': None, 'b': None, 'y': None, 'bsh': None}
        temp = []
        temp1 = []
        temp2 = []

        for j in range(nl):
            temp.append(LineData['R'][j] / (np.power(LineData['R'][j], 2) + np.power(LineData['Xl'][j], 2)))
            temp1.append(-LineData['Xl'][j] / (np.power(LineData['R'][j], 2) + np.power(LineData['Xl'][j], 2)))
            temp2.append(temp[j] + temp1[j] * 1j)  # Y = G + jB
            branch['g'] = temp[:]  # real part: conductance
            branch['b'] = temp1[:]  # imaginary part: susceptance
            branch['y'] = temp2[:]  # complex admittance
            branch['bsh'] = temp3[:]
        del temp
        del temp1
        del temp2
        del temp3

        for j in range(nl):
            k = int(LineData['from'][j])
            m = int(LineData['to'][j])
            Y[k][m] = -(1 / LineData['tap'][j]) * np.exp(-LineData['phi'][j] * 1j) * branch['y'][j]
            Y[m][k] = -(1 / LineData['tap'][j]) * np.exp(LineData['phi'][j] * 1j) * branch['y'][j]
            Y[k][k] = Y[k][k] + ((1 / LineData['tap'][j]) ** 2) * branch['y'][j] + 1j * branch['bsh'][j]
            Y[m][m] = Y[m][m] + branch['y'][j] + 1j * branch['bsh'][j]
            # KVL is used for derivation
            # The impedance of the low voltage side is converted to the high voltage side
            # Tap and transformer ratio have the same meaning ,like 0.978:1

        for i in range(nb):
            Y[i][i] += bus['bsh'][i] * 1j  # Double-check if this needs to be subtracted instead

        G = np.real(Y)
        B = np.imag(Y)

        # ====================================================================================
        # RESOLUTION OF SUBSYSTEM 1 (First Iteration of Newton-Raphson)
        # ====================================================================================
        tolerance = 0.001
        iteration = 0
        max_iterations = 100

        V = np.zeros((nb, 1), dtype=float)
        theta = np.zeros((nb, 1), dtype=float)
        Pk = np.zeros((nb, 1), dtype=float)
        Qk = np.zeros((nb, 1), dtype=float)
        dP = np.zeros((nb, 1), dtype=float)
        dQ = np.zeros((nb, 1), dtype=float)

        for i in range(nb):
            if BusData['type'][i] == 2:  # Slack bus
                V[i] = BusData['voltage'][i]
                theta[i] = BusData['angle'][i]
            elif BusData['type'][i] == 1:  # PV bus
                V[i] = BusData['voltage'][i]
                theta[i] = 0
            else:  # PQ bus
                V[i] = 1
                theta[i] = 0

        # print(BusData['type'])

        # Test convergence
        # Calculate power mismatches

        for k in range(nb):
            # Calculate active (Pk) and reactive (Qk) power injections
            sumP = 0
            sumQ = 0
            for m in range(nb):
                sumP += V[m] * (G[k, m] * np.cos(theta[k] - theta[m]) + B[k, m] * np.sin(theta[k] - theta[m]))
                sumQ += V[m] * (G[k, m] * np.sin(theta[k] - theta[m]) - B[k, m] * np.cos(theta[k] - theta[m]))
            Pk[k] = V[k] * sumP
            Qk[k] = V[k] * sumQ

            # Calculate mismatches dP (active power) and dQ (reactive power)
            if BusData['type'][k] == 2:  # Slack bus: no mismatches in P or Q
                dP[k] = 0
                dQ[k] = 0
            elif BusData['type'][k] == 1:  # PV bus: no Q mismatch
                dP[k] = bus['P'][k] - Pk[k]
                dQ[k] = 0
            else:  # PQ bus: both mismatches exist
                dP[k] = bus['P'][k] - Pk[k]
                dQ[k] = bus['Q'][k] - Qk[k]

        b = np.append(dP.T, dQ.T, axis=1)  # .T Convert column vector to row
        x = np.append(theta, V, axis=0)

        max_mismatch = np.max(np.abs(b))
        JAC = np.zeros((2 * nb, 2 * nb), dtype=float)

        convergence_report = {'iteration': None, 'mismatch': None}
        iteration_list = []
        mismatch_list = []
        iteration_list.append(iteration)
        mismatch_list.append(max_mismatch)
        convergence_report['iteration'] = iteration_list
        convergence_report['mismatch'] = mismatch_list

        while max_mismatch > tolerance and iteration < max_iterations:
            # Build the Jacobian matrix
            for j in range(nl):
                k = int(LineData['from'][j])
                m = int(LineData['to'][j])

                # Submatrix H
                JAC[k][m] = V[k] * V[m] * (
                        G[k, m] * np.sin(theta[k] - theta[m]) - B[k, m] * np.cos(theta[k] - theta[m]))
                JAC[m][k] = V[m] * V[k] * (
                        -G[k, m] * np.sin(theta[k] - theta[m]) - B[k, m] * np.cos(theta[k] - theta[m]))
                #  ‘-’ is because sin(x) = -sin(-x)
                if BusData['type'][k] == 2:  # Slack bus
                    JAC[k][k] = 10 ** 99  # No need to correct the phase angle at the balanced node, let Δδ be zero.
                else:
                    JAC[k][k] = -Qk[k] - (V[k] ** 2 * B[k][k])
                if BusData['type'][k] == 2:
                    JAC[m][m] = 10 ** 99
                else:
                    JAC[m][m] = -Qk[m] - (V[m] ** 2 * B[m][m])

                # Submatrix N
                JAC[k][nb + k] = (Pk[k] + G[k][k] * V[k] ** 2) / V[k]
                JAC[m][nb + m] = (Pk[m] + G[m][m] * V[m] ** 2) / V[m]
                JAC[k][nb + m] = V[k] * (G[k, m] * np.cos(theta[k] - theta[m]) + B[k, m] * np.sin(theta[k] - theta[m]))
                JAC[m][nb + k] = V[m] * (G[k, m] * np.cos(theta[k] - theta[m]) - B[k, m] * np.sin(theta[k] - theta[m]))

                # Submatrix M
                JAC[nb + k][k] = -G[k, k] * V[k] ** 2 + Pk[k]
                JAC[nb + m][m] = -G[m, m] * V[m] ** 2 + Pk[m]
                JAC[nb + k][m] = -V[k] * V[m] * (
                        G[k, m] * np.cos(theta[k] - theta[m]) + B[k, m] * np.sin(theta[k] - theta[m]))
                JAC[nb + m][k] = -V[k] * V[m] * (
                        G[k, m] * np.cos(theta[k] - theta[m]) - B[k, m] * np.sin(theta[k] - theta[m]))

                # Submatrix L
                if BusData['type'][k] == 0:  # PQ bus
                    JAC[nb + k][nb + k] = (Qk[k] - B[k, k] * V[k] ** 2) / V[k]
                else:  # Slack or PV bus
                    JAC[nb + k][nb + k] = 10 ** 99
                # No need to correct the voltage amplitude at the balanced node and PV node, let Δδ,ΔU be zero.
                if BusData['type'][m] == 0:
                    JAC[nb + m][nb + m] = (Qk[m] - B[m, m] * V[m] ** 2) / V[m]
                else:
                    JAC[nb + m][nb + m] = 10 ** 99
                # No need to correct the voltage amplitude at the balanced node and PV node, let Δδ,ΔU be zero.
                JAC[nb + k][nb + m] = V[k] * (
                        G[k, m] * np.sin(theta[k] - theta[m]) - B[k, m] * np.cos(theta[k] - theta[m]))
                JAC[nb + m][nb + k] = -V[m] * (
                        G[k, m] * np.sin(theta[k] - theta[m]) + B[k, m] * np.cos(theta[k] - theta[m]))

            # Solve the system to find state variable corrections (deltaTheta and deltaV)
            dx = np.linalg.solve(JAC, b.T)  # internally uses LU decomposition
            x = x + dx  # x includes the original values of θ and V
            theta = np.copy(x[0:nb])
            V = np.copy(x[nb:2 * nb])

            # Increment the iteration count
            iteration += 1

            # Check if any PQ bus can revert to PV
            for k in range(nb):
                if BusData['Qmax_gen'][k] > 0 and BusData['Qmin_gen'][k] < 0:
                    if V[k] >= BusData['voltage'][k]:
                        V[k] = BusData['voltage'][k]
                        BusData['type'][k] = 1  # Convert back to PV

            # Recalculate Pk and Qk, and update mismatches
            for k in range(nb):
                sumP = 0
                sumQ = 0
                for m in range(nb):
                    sumP += V[m] * (G[k, m] * np.cos(theta[k] - theta[m]) + B[k, m] * np.sin(theta[k] - theta[m]))
                    sumQ += V[m] * (G[k, m] * np.sin(theta[k] - theta[m]) - B[k, m] * np.cos(theta[k] - theta[m]))
                Pk[k] = V[k] * sumP
                Qk[k] = V[k] * sumQ

                if BusData['type'][k] == 2:  # Slack bus
                    dP[k] = 0
                    dQ[k] = 0
                elif BusData['type'][k] == 1:  # PV bus
                    dP[k] = bus['P'][k] - Pk[k]
                    dQ[k] = 0
                else:  # PQ bus
                    dP[k] = bus['P'][k] - Pk[k]
                    dQ[k] = bus['Q'][k] - Qk[k]

            # Check reactive power limits, possibly convert PV to PQ
            for k in range(nb):
                if BusData['type'][k] == 1:
                    if Qk[k] > BusData['Qmax_gen'][k]:
                        bus['Q'][k] = BusData['Qmax_gen'][k]
                        BusData['type'][k] = 0  # Becomes PQ
                    elif Qk[k] < BusData['Qmin_gen'][k]:
                        bus['Q'][k] = BusData['Qmin_gen'][k]
                        BusData['type'][k] = 0  # Becomes PQ

            b = np.zeros((2 * nb, 1), dtype=float)  # b stores dP and dQ, one row
            b = np.append(dP.T, dQ.T, axis=1)
            max_mismatch = np.max(np.abs(b))
            iteration_list.append(iteration)
            mismatch_list.append(max_mismatch)
            convergence_report['iteration'] = iteration_list
            convergence_report['mismatch'] = mismatch_list

        # ====================================================================================
        # RESOLUTION OF SUBSYSTEM 2 (Post-Convergence Calculations)
        # ====================================================================================

        # Compute Pk for slack buses and Qk for slack and PV buses
        for k in range(nb):
            sumP = 0
            sumQ = 0
            for m in range(nb):
                if BusData['type'][k] == 2:
                    sumP += V[m] * (G[k, m] * np.cos(theta[k] - theta[m]) + B[k, m] * np.sin(theta[k] - theta[m]))
                    sumQ += V[m] * (G[k, m] * np.sin(theta[k] - theta[m]) - B[k, m] * np.cos(theta[k] - theta[m]))
                elif BusData['type'][k] == 1:
                    sumQ += V[m] * (G[k, m] * np.sin(theta[k] - theta[m]) - B[k, m] * np.cos(theta[k] - theta[m]))
            if BusData['type'][k] == 2:
                Pk[k] = V[k] * sumP
                Qk[k] = V[k] * sumQ
            elif BusData['type'][k] == 1:
                Pk[k] = BusData['Pgen'][k] / Sbase
                Qk[k] = V[k] * sumQ

        # Compute active/reactive power flows and losses in branches
        Flow = {
            'Pkm': None, 'Pmk': None,
            'Qkm': None, 'Qmk': None,
            'ActiveLoss_km': None, 'ReactiveLoss_km': None
        }

        temp, temp1, temp2, temp3, temp4, temp5 = [], [], [], [], [], []

        for j in range(nl):
            k = int(LineData['from'][j])
            m = int(LineData['to'][j])
            gkm = branch['g'][j]
            bkm = branch['b'][j]
            bkmsh = branch['bsh'][j]  # Line shunt admittance
            akm = LineData['tap'][j]  # Transformer tap ratio
            fikm = LineData['phi'][j]  # Transformer phase shift

            # Active Power Flow from bus k to m, Pkm
            temp.append((((1 / akm) * V[k]) ** 2) * gkm - ((1 / akm) * V[k] * V[m] *
                                                           (gkm * np.cos(theta[k] - theta[m] + fikm) +
                                                            bkm * np.sin(theta[k] - theta[m] + fikm))))
            # Active Power Flow from bus m to k, Pmk
            temp1.append((V[m] ** 2) * gkm - (1 / akm) * V[k] * V[m] *
                         (gkm * np.cos(theta[k] - theta[m] + fikm) -
                          bkm * np.sin(theta[k] - theta[m] + fikm)))
            # Reactive Power Flow from Bus k to m, Qkm
            temp2.append(-((1 / akm * V[k]) ** 2) * (bkm + bkmsh) -
                         (1 / akm) * V[k] * V[m] *
                         (gkm * np.sin(theta[k] - theta[m] + fikm) -
                          bkm * np.cos(theta[k] - theta[m] + fikm)))
            # Reactive Power Flow from Bus m to k, Qmk
            temp3.append(-(V[m] ** 2) * (bkm + bkmsh) +
                         (1 / akm) * V[k] * V[m] *
                         (gkm * np.sin(theta[k] - theta[m] + fikm) +
                          bkm * np.cos(theta[k] - theta[m] + fikm)))
            # Active Power Loss in Branch km
            temp4.append(np.abs(np.abs(temp[j])) - np.abs(temp1[j]))
            # Reactive Power Loss in Branch km
            temp5.append(np.abs(np.abs(temp2[j])) - np.abs(temp3[j]))

        # Flow['Pkm'] = temp[:]
        # Flow['Pmk'] = temp1[:]
        # Flow['Qkm'] = temp2[:]
        # Flow['Qmk'] = temp3[:]
        # Flow['ActiveLoss_km'] = temp4[:]
        # Flow['ReactiveLoss_km'] = temp5[:]

        Flow['Pkm'] = [round(float(val) * Sbase, 3) for val in temp]
        Flow['Pmk'] = [round(float(val) * Sbase, 3) for val in temp1]
        Flow['Qkm'] = [round(float(val) * Sbase, 3) for val in temp2]
        Flow['Qmk'] = [round(float(val) * Sbase, 3) for val in temp3]
        Flow['ActiveLoss_km'] = [round(float(val) * Sbase, 3) for val in temp4]
        Flow['ReactiveLoss_km'] = [round(float(val) * Sbase, 3) for val in temp5]

        del temp, temp1, temp2, temp3, temp4, temp5

        # Convert power values from p.u. to MVA
        Pk *= Sbase
        Qk *= Sbase
        dP *= Sbase
        dQ *= Sbase

        # Convert angles from radians to degrees
        theta_degrees = (theta / np.pi) * 180

        # Store results in dictionary for better readability
        results = {
            'V': None, 'theta_degrees': None, 'theta': None,
            'Pk': None, 'Qk': None, 'dP': None, 'dQ': None
        }

        a, b, c, d, e, f_f, g = [], [], [], [], [], [], []

        for i in range(nb):
            a.append(float(V[i]))
            b.append(float(theta_degrees[i]))
            c.append(float(theta[i]))
            d.append(float(Pk[i]))
            e.append(float(Qk[i]))
            f_f.append(float(dP[i]))
            g.append(float(dQ[i]))
            results['V'] = a[:]
            results['theta_degrees'] = b[:]
            results['theta'] = c[:]
            results['Pk'] = d[:]
            results['Qk'] = e[:]
            results['dP'] = f_f[:]
            results['dQ'] = g[:]
        del a, b, c, d, e, f_f, g

        BusData_1 = {
            'num': None, 'type': None, 'voltage': None, 'angle': None,
            'angle_radian': None, 'Pgen': None, 'Qgen': None, 'Qmin_gen': None, 'Qmax_gen': None,
            'Pload': None, 'Qload': None, 'Qshunt': None, 'hasGen': None,
            'InertiaConst': None, 'Damping': None, 'TransientSusceptance': None,
            'Pk': None, 'Qk': None
        }

        temp_voltage = results['V']
        temp_degrees = results['theta_degrees']
        temp_rad = results['theta']
        temp_Pk = results['Pk']
        temp_Qk = results['Qk']

        for key in BusData.keys():
            temp = []
            if key == 'voltage':
                temp = results['V']
            elif key == 'angle':
                temp = results['theta_degrees']
            else:
                temp = BusData[key]
            BusData_1[key] = temp[:]
            del temp

        BusData_1['angle_radian'] = temp_rad
        BusData_1['Pk'] = temp_Pk
        BusData_1['Qk'] = temp_Qk

        # df = pd.DataFrame(BusData_1)
        # df.to_excel('BusData_2.xlsx', index=False)

        for key in Flow.keys():
            temp = Flow[key]
            LineData_1[key] = temp[:]
            del temp

        # df = pd.DataFrame(LineData_1)
        # df.to_excel('LineData_1.xlsx', index=False)

        return BusData_1, LineData_1

    @staticmethod
    def rec_power_flow(Sbase, Ubase, f, BusData, LineData):
        Zbase = (Ubase ** 2) / Sbase

        # Transforms transient reactance X' into transient admittance Y'

        for i in range(len(BusData['TransientSusceptance'])):
            if BusData['TransientSusceptance'][i] != 0:
                BusData['TransientSusceptance'][i] = 1 / BusData['TransientSusceptance'][i]

        # Save Line output
        # Create dictionary from line data list
        LineData_1 = {
            'from': None, 'to': None, 'R': None, 'Xl': None, 'Xc': None,
            'tap': None, 'tapmin': None, 'tapmax': None, 'phi': None,
            'Pkm': None, 'Pmk': None, 'Qkm': None, 'Qmk': None,
            'ActiveLoss_km': None, 'ReactiveLoss_km': None
        }

        for key in LineData.keys():
            temp = LineData[key]
            LineData_1[key] = temp[:]
            del temp

        # Adjust bus numbers for Python indexing (start from 0)
        for k in range(len(LineData['from'])):
            LineData['from'][k] -= 1
        for k in range(len(LineData['to'])):
            LineData['to'][k] -= 1

        # ---------------------------------------------------------------------
        # Extract system size
        nb = len(BusData['num'])  # Number of buses
        nl = len(LineData['from'])  # Number of lines
        npv = 0
        npq = 0
        for i in range(nb):
            if BusData['type'][i] == 1:
                npv += 1
            elif BusData['type'][i] == 0:
                npq += 1

        # ---------------------------------------------------------------------
        # Normalize and compute net power for each bus

        temp3 = []
        temp4 = []
        bus = {'P': None, 'Q': None, 'bsh': None}
        for i in range(nb):
            temp3.append((BusData['Pgen'][i] - BusData['Pload'][i]) / Sbase)  # calculate pu value
            temp4.append((BusData['Qgen'][i] - BusData['Qload'][i]) / Sbase)  # calculate pu value
            BusData['Pload'][i] = BusData['Pload'][i] / Sbase  # calculate pu value
            BusData['Qload'][i] = BusData['Qload'][i] / Sbase  # calculate pu value
            bus['P'] = temp3[:]
            bus['Q'] = temp4[:]
        del temp3
        del temp4

        temp5 = []
        for i in range(nb):
            BusData['voltage'][i] = BusData['voltage'][i] / Ubase  # calculate voltage pu value
            BusData['angle'][i] = BusData['angle'][i] * (np.pi / 180)  # Convert degrees to radians
            BusData['Qshunt'][i] = BusData['Qshunt'][i] / Sbase  # calculate Q pu value
            temp5.append(BusData['Qshunt'][i])
            bus['bsh'] = temp5[:]
        del temp5

        # those two lines can save BusData
        # df = pd.DataFrame(BusData)
        # df.to_excel('BusData.xlsx', index=False)

        # ---------------------------------------------------------------------
        # Convert line parameters to per-unit (pu) system

        for i in range(nl):  # Transformer taps 1 means voltage unchanged
            if LineData['tap'][i] == 0:
                LineData['tap'][i] = 1

        num_tap_normal = 0
        num_phase_shift = 0
        temp3 = []
        for i in range(nl):
            LineData['R'][i] = LineData['R'][i] / Zbase  # ohm to pu
            LineData['Xl'][i] = (LineData['Xl'][i] * 2 * np.pi * f) / Zbase
            LineData['Xc'][i] = (LineData['Xc'][i] * 2 * np.pi * f * Zbase) / 2
            temp3.append(LineData['Xc'][i])  # Calculate Lsh
            LineData['phi'][i] = LineData['phi'][i] * (np.pi / 180)  # Phase-shifter angle to radians
            if LineData['tap'][i] != 1.0:
                num_tap_normal += 1
            elif LineData['phi'][i] != 0.0:
                num_phase_shift += 1

        # ---------------------------------------------------------------------
        # Build Y, G, B matrices for power flow analysis

        Y = np.zeros((nb, nb), dtype=complex)
        branch = {'g': None, 'b': None, 'y': None, 'bsh': None}
        temp = []
        temp1 = []
        temp2 = []

        for j in range(nl):
            temp.append(LineData['R'][j] / (np.power(LineData['R'][j], 2) + np.power(LineData['Xl'][j], 2)))
            temp1.append(-LineData['Xl'][j] / (np.power(LineData['R'][j], 2) + np.power(LineData['Xl'][j], 2)))
            temp2.append(temp[j] + temp1[j] * 1j)  # Y = G + jB
            branch['g'] = temp[:]  # real part: conductance
            branch['b'] = temp1[:]  # imaginary part: susceptance
            branch['y'] = temp2[:]  # complex admittance
            branch['bsh'] = temp3[:]
        del temp
        del temp1
        del temp2
        del temp3

        for j in range(nl):
            k = int(LineData['from'][j])
            m = int(LineData['to'][j])
            Y[k][m] = -(1 / LineData['tap'][j]) * np.exp(-LineData['phi'][j] * 1j) * branch['y'][j]
            Y[m][k] = -(1 / LineData['tap'][j]) * np.exp(LineData['phi'][j] * 1j) * branch['y'][j]
            Y[k][k] = Y[k][k] + ((1 / LineData['tap'][j]) ** 2) * branch['y'][j] + 1j * branch['bsh'][j]
            Y[m][m] = Y[m][m] + branch['y'][j] + 1j * branch['bsh'][j]
            # KVL is used for derivation
            # The impedance of the low voltage side is converted to the high voltage side
            # Tap and transformer ratio have the same meaning ,like 0.978:1

        for i in range(nb):
            Y[i][i] += bus['bsh'][i] * 1j  # Double-check if this needs to be subtracted instead

        G = np.real(Y)
        B = np.imag(Y)

        # ====================================================================================
        # RESOLUTION OF SUBSYSTEM 1 (First Iteration of Newton-Raphson)
        # ====================================================================================

        tolerance = 0.001
        iteration = 0
        max_iterations = 100

        # Initialize voltage real (e) and imaginary (f) parts
        V = np.zeros((nb, 1), dtype=float)
        theta = np.zeros((nb, 1), dtype=float)
        e = np.zeros((nb, 1), dtype=float)
        f = np.zeros((nb, 1), dtype=float)
        Pk = np.zeros((nb, 1), dtype=float)
        Qk = np.zeros((nb, 1), dtype=float)
        dP = np.zeros((nb, 1), dtype=float)
        dQ = np.zeros((nb, 1), dtype=float)

        for i in range(nb):
            V[i] = BusData['voltage'][i]
            theta[i] = BusData['angle'][i]

            # Slack and PV buses use given voltage magnitude
            if BusData['type'][i] == 2 or BusData['type'][i] == 1:
                V[i] = BusData['voltage'][i]
                e[i] = V[i] * np.cos(theta[i])
                f[i] = V[i] * np.sin(theta[i])
            else:  # PQ bus: assume flat start (V = 1∠0)
                V[i] = 1.0
                e[i] = 1.0
                f[i] = 0.0

        print(BusData['type'])

        # Test convergence
        # Calculate power mismatches

        for k in range(nb):
            sum1 = 0.0
            sum2 = 0.0
            for m in range(nb):
                sum1 += G[k, m] * e[m] - B[k, m] * f[m]
                sum2 += G[k, m] * f[m] + B[k, m] * e[m]
            Pk[k] = e[k] * sum1 + f[k] * sum2
            Qk[k] = f[k] * sum1 - e[k] * sum2

            # Mismatch calculation
            if BusData['type'][k] == 2:  # Slack bus
                dP[k] = 0
                dQ[k] = 0
            elif BusData['type'][k] == 1:  # PV bus: no Q mismatch
                dP[k] = bus['P'][k] - Pk[k]
                dQ[k] = V[k] ** 2 - e[k] ** 2 - f[k] ** 2
            else:  # PQ bus
                dP[k] = bus['P'][k] - Pk[k]
                dQ[k] = bus['Q'][k] - Qk[k]

        b = np.append(dP.T, dQ.T, axis=1)  # .T Convert column vector to row
        x = np.append(f, e, axis=0)  # state vector is [e; f]

        max_mismatch = np.max(np.abs(b))
        JAC = np.zeros((2 * nb, 2 * nb), dtype=float)

        convergence_report = {'iteration': None, 'mismatch': None}
        iteration_list = []
        mismatch_list = []
        iteration_list.append(iteration)
        mismatch_list.append(max_mismatch)
        convergence_report['iteration'] = iteration_list
        convergence_report['mismatch'] = mismatch_list

        def gaussianelimination(J, dPQ_minus):  # Full pivoting
            n = len(dPQ_minus)
            A = copy.deepcopy(J)
            b = copy.deepcopy(dPQ_minus)

            col_order = list(range(n))  # Track column swaps

            # Forward Elimination with Full Pivoting
            for k in range(n):
                # Find pivot in submatrix A[k:][k:]
                max_val = 0
                pivot_row = pivot_col = k
                for i in range(k, n):
                    for j in range(k, n):
                        if abs(A[i][j]) > abs(max_val):
                            max_val = A[i][j]
                            pivot_row = i
                            pivot_col = j

                if abs(max_val) < 1e-12:
                    raise ValueError("Matrix is singular or nearly singular.")

                # Swap rows
                A[k], A[pivot_row] = A[pivot_row], A[k]
                b[k], b[pivot_row] = b[pivot_row], b[k]

                # Swap columns (track permutations)
                for row in A:
                    row[k], row[pivot_col] = row[pivot_col], row[k]
                col_order[k], col_order[pivot_col] = col_order[pivot_col], col_order[k]

                # Eliminate entries below pivot
                for i in range(k + 1, n):
                    factor = A[i][k] / A[k][k]
                    for j in range(k, n):
                        A[i][j] -= factor * A[k][j]
                    b[i] -= factor * b[k]

            # Back Substitution
            x = [0.0] * n
            for i in reversed(range(n)):
                sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))
                x[i] = (b[i] - sum_ax) / A[i][i]

            # Undo column permutations to restore original order
            x_final = [0.0] * n
            for i in range(n):
                x_final[col_order[i]] = x[i]

            return x_final

        while max_mismatch > tolerance and iteration < max_iterations:

            for j in range(nl):
                k = int(LineData['from'][j])
                m = int(LineData['to'][j])

                Gkm = G[k, m]
                Bkm = B[k, m]

                # ==============================
                # Submatrix H (∂P/∂f)
                # ==============================
                JAC[k][m] += -Bkm * e[k] + Gkm * f[k]
                JAC[m][k] += -Bkm * e[m] + Gkm * f[m]

                # ==============================
                # Submatrix N (∂P/∂e)
                # ==============================
                JAC[k][nb + m] += Gkm * e[k] + Bkm * f[k]
                JAC[m][nb + k] += Gkm * e[m] + Bkm * f[m]

                # ==============================
                # Submatrix M (∂Q/∂f)
                # ==============================
                if BusData['type'][k] == 0:  # PQ bus
                    JAC[nb + k][m] += -Gkm * e[k] - Bkm * f[k]
                elif BusData['type'][k] == 1:  # PV bus (R): keep 0
                    JAC[nb + k][m] += 0

                if BusData['type'][m] == 0:
                    JAC[nb + m][k] += -Gkm * e[m] - Bkm * f[m]
                elif BusData['type'][m] == 1:
                    JAC[nb + m][k] += 0

                # ==============================
                # Submatrix L (∂Q/∂e)
                # ==============================
                if BusData['type'][k] == 0:  # PQ bus
                    JAC[nb + k][nb + m] += -Bkm * e[k] + Gkm * f[k]
                elif BusData['type'][k] == 1:  # PV bus (S): keep 0
                    JAC[nb + k][nb + m] += 0

                if BusData['type'][m] == 0:
                    JAC[nb + m][nb + k] += -Bkm * e[m] + Gkm * f[m]
                elif BusData['type'][m] == 1:
                    JAC[nb + m][nb + k] += 0

            for k in range(nb):
                sumG_e = 0.0
                sumG_f = 0.0
                sumB_e = 0.0
                sumB_f = 0.0

                for m in range(nb):
                    sumG_e += G[k, m] * e[m]
                    sumG_f += G[k, m] * f[m]
                    sumB_e += B[k, m] * e[m]
                    sumB_f += B[k, m] * f[m]

                ai = sumG_e - sumB_f
                bi = sumG_f + sumB_e

                # H: ∂P_k/∂f_k
                JAC[k][k] = -B[k, k] * e[k] + G[k, k] * f[k] + bi

                # N: ∂P_k/∂e_k
                JAC[k][nb + k] = G[k, k] * e[k] + B[k, k] * f[k] + ai

                # M: ∂Q_k/∂f_k
                if BusData['type'][k] == 0:  # PQ bus
                    JAC[nb + k][k] = -G[k, k] * e[k] - B[k, k] * f[k] + ai
                elif BusData['type'][k] == 1:  # PV bus: use R
                    JAC[nb + k][k] = -2 * f[k]  # R term

                # L: ∂Q_k/∂e_k
                if BusData['type'][k] == 0:  # PQ bus
                    JAC[nb + k][nb + k] = -B[k, k] * e[k] + G[k, k] * f[k] - bi
                elif BusData['type'][k] == 1:  # PV bus: use S
                    JAC[nb + k][nb + k] = -2 * e[k]  # S term

                if BusData['type'][k] == 2:  # Slack bus
                    JAC[k][k] = 10 ** 99  # lock Δe_k
                    JAC[nb + k][nb + k] = 10 ** 99  # lock Δf_k

            # Solve the system to find state variable corrections (delta_e and delta_f)
            dx = np.linalg.solve(JAC, b.T)  # Solves: JAC * dx = b
            x = x + dx

            # Update state variables: x = [e; f]
            e = np.copy(x[0:nb])
            f = np.copy(x[nb:2 * nb])

            # Increment iteration counter
            iteration += 1

            # Check if any PQ bus can revert to PV
            for k in range(nb):
                if BusData['Qmax_gen'][k] > 0 and BusData['Qmin_gen'][k] < 0:
                    V_mag_k = np.sqrt(e[k] ** 2 + f[k] ** 2)
                    if V_mag_k >= BusData['voltage'][k]:
                        # Clamp to specified voltage magnitude
                        scale = BusData['voltage'][k] / V_mag_k
                        e[k] *= scale
                        f[k] *= scale
                        BusData['type'][k] = 1  # Convert back to PV

            # Recalculate power injections and mismatches
            for k in range(nb):
                sum1 = 0.0
                sum2 = 0.0
                for m in range(nb):
                    sum1 += G[k, m] * e[m] - B[k, m] * f[m]
                    sum2 += G[k, m] * f[m] + B[k, m] * e[m]
                Pk[k] = e[k] * sum1 + f[k] * sum2
                Qk[k] = f[k] * sum1 - e[k] * sum2

                # Mismatch
                if BusData['type'][k] == 2:  # Slack
                    dP[k] = 0
                    dQ[k] = 0
                elif BusData['type'][k] == 1:  # PV
                    dP[k] = bus['P'][k] - Pk[k]
                    dQ[k] = 0
                else:  # PQ
                    dP[k] = bus['P'][k] - Pk[k]
                    dQ[k] = bus['Q'][k] - Qk[k]

            # Check Q limits and downgrade PV to PQ
            for k in range(nb):
                if BusData['type'][k] == 1:
                    if Qk[k] > BusData['Qmax_gen'][k]:
                        bus['Q'][k] = BusData['Qmax_gen'][k]
                        BusData['type'][k] = 0  # becomes PQ
                    elif Qk[k] < BusData['Qmin_gen'][k]:
                        bus['Q'][k] = BusData['Qmin_gen'][k]
                        BusData['type'][k] = 0  # becomes PQ

            # Rebuild mismatch vector
            b = np.zeros((2 * nb, 1), dtype=float)  # b stores dP and dQ, one row
            b = np.append(dP.T, dQ.T, axis=1)  # Shape: 1 × (2*nb)
            max_mismatch = np.max(np.abs(b))
            # Log convergence
            iteration_list.append(iteration)
            mismatch_list.append(max_mismatch)
            convergence_report['iteration'] = iteration_list
            convergence_report['mismatch'] = mismatch_list

        # ====================================================================================
        # RESOLUTION OF SUBSYSTEM 2 (Post-Convergence Calculations)
        # ====================================================================================

        # ===============================
        # Convert bus voltage to polar form
        # ===============================

        for k in range(nb):
            V[k] = np.sqrt(e[k] ** 2 + f[k] ** 2)
            theta[k] = np.arctan2(f[k], e[k])

        for k in range(nb):
            sumP = 0
            sumQ = 0
            for m in range(nb):
                if BusData['type'][k] == 2:
                    sumP += V[m] * (G[k, m] * np.cos(theta[k] - theta[m]) + B[k, m] * np.sin(theta[k] - theta[m]))
                    sumQ += V[m] * (G[k, m] * np.sin(theta[k] - theta[m]) - B[k, m] * np.cos(theta[k] - theta[m]))
                elif BusData['type'][k] == 1:
                    sumQ += V[m] * (G[k, m] * np.sin(theta[k] - theta[m]) - B[k, m] * np.cos(theta[k] - theta[m]))
            if BusData['type'][k] == 2:
                Pk[k] = V[k] * sumP
                Qk[k] = V[k] * sumQ
            elif BusData['type'][k] == 1:
                Pk[k] = BusData['Pgen'][k] / Sbase
                Qk[k] = V[k] * sumQ

        # Compute active/reactive power flows and losses in branches
        Flow = {
            'Pkm': None, 'Pmk': None,
            'Qkm': None, 'Qmk': None,
            'ActiveLoss_km': None, 'ReactiveLoss_km': None
        }

        temp, temp1, temp2, temp3, temp4, temp5 = [], [], [], [], [], []

        for j in range(nl):
            k = int(LineData['from'][j])
            m = int(LineData['to'][j])
            gkm = branch['g'][j]
            bkm = branch['b'][j]
            bkmsh = branch['bsh'][j]  # Line shunt admittance
            akm = LineData['tap'][j]  # Transformer tap ratio
            fikm = LineData['phi'][j]  # Transformer phase shift

            # Active Power Flow from bus k to m, Pkm
            temp.append((((1 / akm) * V[k]) ** 2) * gkm - ((1 / akm) * V[k] * V[m] *
                                                           (gkm * np.cos(theta[k] - theta[m] + fikm) +
                                                            bkm * np.sin(theta[k] - theta[m] + fikm))))
            # Active Power Flow from bus m to k, Pmk
            temp1.append((V[m] ** 2) * gkm - (1 / akm) * V[k] * V[m] *
                         (gkm * np.cos(theta[k] - theta[m] + fikm) -
                          bkm * np.sin(theta[k] - theta[m] + fikm)))
            # Reactive Power Flow from Bus k to m, Qkm
            temp2.append(-((1 / akm * V[k]) ** 2) * (bkm + bkmsh) -
                         (1 / akm) * V[k] * V[m] *
                         (gkm * np.sin(theta[k] - theta[m] + fikm) -
                          bkm * np.cos(theta[k] - theta[m] + fikm)))
            # Reactive Power Flow from Bus m to k, Qmk
            temp3.append(-(V[m] ** 2) * (bkm + bkmsh) +
                         (1 / akm) * V[k] * V[m] *
                         (gkm * np.sin(theta[k] - theta[m] + fikm) +
                          bkm * np.cos(theta[k] - theta[m] + fikm)))
            # Active Power Loss in Branch km
            temp4.append(np.abs(np.abs(temp[j])) - np.abs(temp1[j]))
            # Reactive Power Loss in Branch km
            temp5.append(np.abs(np.abs(temp2[j])) - np.abs(temp3[j]))

        # Flow['Pkm'] = temp[:]
        # Flow['Pmk'] = temp1[:]
        # Flow['Qkm'] = temp2[:]
        # Flow['Qmk'] = temp3[:]
        # Flow['ActiveLoss_km'] = temp4[:]
        # Flow['ReactiveLoss_km'] = temp5[:]

        Flow['Pkm'] = [round(float(val) * Sbase, 3) for val in temp]
        Flow['Pmk'] = [round(float(val) * Sbase, 3) for val in temp1]
        Flow['Qkm'] = [round(float(val) * Sbase, 3) for val in temp2]
        Flow['Qmk'] = [round(float(val) * Sbase, 3) for val in temp3]
        Flow['ActiveLoss_km'] = [round(float(val) * Sbase, 3) for val in temp4]
        Flow['ReactiveLoss_km'] = [round(float(val) * Sbase, 3) for val in temp5]

        del temp, temp1, temp2, temp3, temp4, temp5

        # Convert power values from p.u. to MVA
        Pk *= Sbase
        Qk *= Sbase
        dP *= Sbase
        dQ *= Sbase

        # Convert angles from radians to degrees
        theta_degrees = (theta / np.pi) * 180

        # Store results in dictionary for better readability
        results = {
            'V': None, 'theta_degrees': None, 'theta': None,
            'Pk': None, 'Qk': None, 'dP': None, 'dQ': None
        }

        a, b, c, d, e, f_f, g = [], [], [], [], [], [], []

        for i in range(nb):
            a.append(float(V[i]))
            b.append(float(theta_degrees[i]))
            c.append(float(theta[i]))
            d.append(float(Pk[i]))
            e.append(float(Qk[i]))
            f_f.append(float(dP[i]))
            g.append(float(dQ[i]))
            results['V'] = a[:]
            results['theta_degrees'] = b[:]
            results['theta'] = c[:]
            results['Pk'] = d[:]
            results['Qk'] = e[:]
            results['dP'] = f_f[:]
            results['dQ'] = g[:]
        del a, b, c, d, e, f_f, g

        BusData_1 = {
            'num': None, 'type': None, 'voltage': None, 'angle': None,
            'angle_radian': None, 'Pgen': None, 'Qgen': None, 'Qmin_gen': None, 'Qmax_gen': None,
            'Pload': None, 'Qload': None, 'Qshunt': None, 'hasGen': None,
            'InertiaConst': None, 'Damping': None, 'TransientSusceptance': None,
            'Pk': None, 'Qk': None
        }

        temp_voltage = results['V']
        temp_degrees = results['theta_degrees']
        temp_rad = results['theta']
        temp_Pk = results['Pk']
        temp_Qk = results['Qk']

        for key in BusData.keys():
            temp = []
            if key == 'voltage':
                temp = results['V']
            elif key == 'angle':
                temp = results['theta_degrees']
            else:
                temp = BusData[key]
            BusData_1[key] = temp[:]
            del temp

        BusData_1['angle_radian'] = temp_rad
        BusData_1['Pk'] = temp_Pk
        BusData_1['Qk'] = temp_Qk

        # df = pd.DataFrame(BusData_1)
        # df.to_excel('BusData_2.xlsx', index=False)

        for key in Flow.keys():
            temp = Flow[key]
            LineData_1[key] = temp[:]
            del temp

        # df = pd.DataFrame(LineData_1)
        # df.to_excel('LineData_1.xlsx', index=False)

        return BusData_1, LineData_1

    @staticmethod
    def step_by_step(Sbase, Ubase, f, BusData, LineData, CBus, Line, frombus, tobus, fstart, fend, Simtime, mstep):
        Sbase = 100
        Ubase = 138
        Zbase = (Ubase ** 2) / Sbase
        # f = 60

        # Adjust bus numbers for Python indexing (start from 0)
        for k in range(len(LineData['from'])):
            LineData['from'][k] -= 1
        for k in range(len(LineData['to'])):
            LineData['to'][k] -= 1

        # ---------------------------------------------------------------------
        # Extract system size
        nb = len(BusData['num'])  # Number of buses
        nl = len(LineData['from'])  # Number of lines
        npv = 0
        npq = 0
        for i in range(nb):
            if BusData['type'][i] == 1:
                npv += 1
            elif BusData['type'][i] == 0:
                npq += 1

        for i in range(nl):  # Transformer taps 1 means voltage unchanged
            if LineData['tap'][i] == 0:
                LineData['tap'][i] = 1

        # ---------------------------------------------------------------------
        # Normalize and compute net power for each bus

        temp3 = []
        temp4 = []
        bus = {'P': None, 'Q': None, 'bsh': None}
        for i in range(nb):
            temp3.append((BusData['Pgen'][i] - BusData['Pload'][i]) / Sbase)  # calculate pu value
            temp4.append((BusData['Qgen'][i] - BusData['Qload'][i]) / Sbase)  # calculate pu value
            bus['P'] = temp3[:]
            bus['Q'] = temp4[:]
        del temp3
        del temp4

        temp5 = []
        for i in range(nb):
            temp5.append(BusData['Qshunt'][i])
            bus['bsh'] = temp5[:]
        del temp5

        # those two lines can save BusData
        # df = pd.DataFrame(BusData)
        # df.to_excel('BusData.xlsx', index=False)

        # ---------------------------------------------------------------------
        # Convert line parameters to per-unit (pu) system

        for i in range(nl):  # Transformer taps 1 means voltage unchanged
            if LineData['tap'][i] == 0:
                LineData['tap'][i] = 1

        num_tap_normal = 0
        num_phase_shift = 0
        temp3 = []
        for i in range(nl):
            LineData['R'][i] = LineData['R'][i] / Zbase  # ohm to pu
            LineData['Xl'][i] = (LineData['Xl'][i] * 2 * np.pi * f) / Zbase
            LineData['Xc'][i] = (LineData['Xc'][i] * 2 * np.pi * f * Zbase) / 2
            temp3.append(LineData['Xc'][i])  # Calculate Lsh
            LineData['phi'][i] = LineData['phi'][i] * (np.pi / 180)  # Phase-shifter angle to radians
            if LineData['tap'][i] != 1.0:
                num_tap_normal += 1
            elif LineData['phi'][i] != 0.0:
                num_phase_shift += 1

        # ---------------------------------------------------------------------
        # Build Y, G, B matrices for power flow analysis

        Ybus = np.zeros((nb, nb), dtype=complex)
        branch = {'g': None, 'b': None, 'y': None, 'bsh': None}
        temp = []
        temp1 = []
        temp2 = []

        for j in range(nl):
            temp.append(LineData['R'][j] / (np.power(LineData['R'][j], 2) + np.power(LineData['Xl'][j], 2)))
            temp1.append(-LineData['Xl'][j] / (np.power(LineData['R'][j], 2) + np.power(LineData['Xl'][j], 2)))
            temp2.append(temp[j] + temp1[j] * 1j)  # Y = G + jB
            branch['g'] = temp[:]  # real part: conductance
            branch['b'] = temp1[:]  # imaginary part: susceptance
            branch['y'] = temp2[:]  # complex admittance
            branch['bsh'] = temp3[:]
        del temp
        del temp1
        del temp2
        del temp3

        for j in range(nl):
            k = int(LineData['from'][j])
            m = int(LineData['to'][j])
            Ybus[k][m] = -(1 / LineData['tap'][j]) * np.exp(-LineData['phi'][j] * 1j) * branch['y'][j]
            Ybus[m][k] = -(1 / LineData['tap'][j]) * np.exp(LineData['phi'][j] * 1j) * branch['y'][j]
            Ybus[k][k] = Ybus[k][k] + ((1 / LineData['tap'][j]) ** 2) * branch['y'][j] + 1j * branch['bsh'][j]
            Ybus[m][m] = Ybus[m][m] + branch['y'][j] + 1j * branch['bsh'][j]
            # KVL is used for derivation
            # The impedance of the low voltage side is converted to the high voltage side
            # Tap and transformer ratio have the same meaning ,like 0.978:1

        for i in range(nb):
            Ybus[i][i] += bus['bsh'][i] * 1j  # Double-check if this needs to be subtracted instead

        G = np.real(Ybus)
        B = np.imag(Ybus)

        V = np.array(BusData['voltage'], dtype=float).reshape(-1, 1)
        theta = np.array(BusData['angle_radian'], dtype=float).reshape(-1, 1)
        Pk = np.array(BusData['Pk'], dtype=float).reshape(-1, 1)

        # file = open(
        #     'D:\\ProgramData\\PycharmProjects\\TransientEn_standard\\IEEE14_standard.txt', 'r')
        # bus, branch, Sbase, theta_degrees, V, theta, Pk = PowerFlow.run_power_flow(file)
        # BusData: Busbar raw data, dictionary, pu
        # LineData: Line raw data, dictionary, pu
        # bus: Contains P net (pu), Q net (pu), bus shunt susceptance (pu)
        # branch: Based on the impedance value of LineData, obtain G, B, Y (pu), half line admittance (pu)
        # Sbase: S base value
        # nb: Bus number
        # nl: Line number
        # Ybus: Bus admittance matrix
        # B: Ybus imaginary part
        # G: Ybus Real part
        # theta_degrees: Bus voltage angle degrees
        # V: Bus voltage amplitude
        # theta: Bus voltage radian
        # Pk: Bus active power, unit MW
        # #########################################################################################################################
        #                                         BEGINNING OF PROGRAM
        # #########################################################################################################################

        # df = pd.DataFrame(BusData)
        # df.to_excel('BusData_StepByStep_1.xlsx', index=False)

        ContingencyBus = CBus  # Indicates the bus where the contingency occurs
        line = Line  # Defines which line will be removed after the contingency
        from_bus = frombus
        to_bus = tobus

        # Variables for numerical integration

        tpre0 = 0.0
        tfout0 = fstart  # Initial fault time
        tfoult = fend  # Fault clearing time (in seconds)
        tAfter = Simtime  # Maximum post-fault simulation time
        maxstep = mstep  # Integration time step

        # --------------------------------------------------------------------------------------------------------------------------
        #                                   STEP: PRE-FAULT REDUCED MATRIX
        # --------------------------------------------------------------------------------------------------------------------------

        # Build matrix Yl

        ng = 0
        TransientAdmittance = []

        print(BusData)
        # Build matrix Y: diagonal of generator transient admittances (Matrix A)
        for i in range(len(BusData['TransientSusceptance'])):  # Count number of generators
            if BusData['hasGen'][i] != 0:
                TransientAdmittance.append(-1j * BusData['TransientSusceptance'][i])
                ng += 1
        # BusData['TransientReactance'] actually stores the magnitude of the generator's transient susceptance.
        # This has been implemented in LibraryReadData.py:
        # LibraryReadData.py-Line 94                 BusData['TransientReactance'][i] = 1 / BusData['TransientReactance'][i]

        Y = np.zeros((ng, ng), dtype=complex)  # Diagonal matrix of generator transient admittances (Matrix A)
        TransientAdmittance = np.array(TransientAdmittance)
        np.fill_diagonal(Y, TransientAdmittance)  # Fill the diagonal elements of matrix Y
        # Y=diagonal(TransientAdmittance), Yii=-JB', which is magnitude of generator transient susceptance
        print(Y)
        print('=-' * 60)
        print(type(TransientAdmittance))

        # Build the constant load admittance vector (Matrix D)
        S_conj = np.zeros(nb, dtype=complex)  # Conjugate of apparent power
        for i in range(len(BusData['num'])):
            S_conj[i] = BusData['Pload'][i] + 1j * BusData['Qload'][i]

        S_conj = np.conjugate(S_conj)  # Take the complex conjugate of each element of S_conj
        Y_const = np.zeros(len(BusData['num']), dtype=complex)  # Load modeled as constant admittance

        for i in range(len(Y_const)):
            Y_const[i] = S_conj[i] / V[i] ** 2
            print(Y_const[i])
        # Meaning: approximate each load as a constant admittance

        # Add load admittance to YbusLPre (Matrix D)
        YbusLPre = np.copy(Ybus)  # Ybus is the node admittance matrix, pu
        for i in range(len(BusData['num'])):
            for j in range(len(BusData['num'])):
                if i == j:
                    print('entered')
                    YbusLPre[i][j] += Y_const[i]

        # Modify YbusLPre to include generator transient admittance (Matrix D)
        YbusLPre_IVg = np.copy(YbusLPre)
        gen_index = 0  # Generator index
        for i in range(len(BusData['num'])):
            if BusData['hasGen'][i] != 0:
                gen_index = i
                YbusLPre[gen_index, gen_index] += -1j * BusData['TransientSusceptance'][i]

        # Build menosY (Matrix B: generator-to-bus coupling)
        menosY = np.zeros((len(Y), len(BusData['num'])), dtype=complex)  # Matrix B (transpose becomes matrix C)
        generator_buses = []

        for i in range(len(BusData['num'])):
            if BusData['hasGen'][i] != 0:
                generator_buses.append(BusData['num'][i] - 1)

        for i in range(len(Y)):
            for j in range(len(BusData['num'])):
                if j == generator_buses[i]:
                    menosY[i, j] = 1j * BusData['TransientSusceptance'][j]

        # Construct complete matrix
        Y1Pre = np.hstack((Y, menosY))
        Y2Pre = np.hstack((menosY.T, YbusLPre))
        YcompletePre = np.vstack((Y1Pre, Y2Pre))
        # [Igen, Ibus] = YcompletePre * [E', V]
        # Igen = Y×E' + menosY×V
        # Ibus = menosY.T×E' + YbusLPre×V

        # Build reduced matrix for pre-fault
        YbusLinvPre = np.linalg.inv(YbusLPre)  # Matrix D
        YredPre = Y - np.dot(np.dot(menosY, YbusLinvPre), menosY.T)  # Reduced matrix before the fault
        # Need Igen = YredPre×E', YredPre = Y - (menosY)×(YcompletePre.inverse)×(menosY.T)
        # Elementary block transformations of matrices P102

        # -------------------------------------------------------------------------------------------------------------------
        #                              STEP: REDUCED MATRIX DURING THE FAULT
        # -------------------------------------------------------------------------------------------------------------------

        # ContingencyBus = 1  # The bus where the contingency occurs (Note: Python indexing starts from 0)

        # Calculate Yred during the fault for the given contingency

        YbusLFault = np.copy(YbusLPre)  # pre fault Ybus load admittance, generator transient added
        menosYFault = np.copy(menosY)  # pre fault generator-to-bus coupling admittance added

        # zero out all elements in the row and column corresponding to the faulted bus in the admittance matrix
        for i in range(len(BusData['num'])):
            for j in range(len(BusData['num'])):
                YbusLFault[i][ContingencyBus - 1] = 0
                YbusLFault[ContingencyBus - 1][j] = 0

        # set the self-admittance (diagonal element) of the faulted bus to 1 pu
        for i in range(len(BusData['num'])):
            for j in range(len(BusData['num'])):
                if i == ContingencyBus - 1 and j == ContingencyBus - 1:
                    YbusLFault[i][j] = 1

        # breaks connection for the corresponding generator
        # find nonzero element in the column corresponding to the contingency bus in menosYFault, set that element to 0.
        # once done, break out of the loops immediately.
        flag = 0
        for i in range(ng):
            for j in range(len(BusData['num'])):
                if j == ContingencyBus - 1:
                    if menosYFault[i][j] != 0:
                        menosYFault[i][j] = 0
                        flag = 1
                    if flag == 1:
                        break
            if flag == 1:
                break

        # Build complete matrix for the fault period

        Y1Fault = np.hstack((Y, menosYFault))
        Y2Fault = np.hstack((menosYFault.T, YbusLFault))
        YcompleteFault = np.vstack((Y1Fault, Y2Fault))

        # Build reduced matrix for fault period

        YbusLinvFault = np.linalg.inv(YbusLFault)  # Matrix D
        YredFault = Y - np.dot(np.dot(menosYFault, YbusLinvFault), menosYFault.T)

        # ------------------------------------------------------------------------------------------------------------------
        #                              STEP: REDUCED MATRIX POST-FAULT
        # ------------------------------------------------------------------------------------------------------------------

        # operation after the fault is to first copy the admittance matrix prefault
        # remove a line after the fault, and modify the admittance matrix according to the removed line
        YbusLPost = np.copy(YbusLPre)
        menosYPost = np.copy(menosY)

        # Remove the affected line (REMEMBER: Python indexes from 0)

        portion_to_remove = complex(1 + 1j)
        for i in range(nl):
            if i == line - 1:
                portion_to_remove = branch['y'][i]  # Store the line admittance to be removed from Ybus
        print(portion_to_remove)

        # Remove the corresponding portion from YbusLPost

        for i in range(nb):
            for j in range(nb):
                if i == ContingencyBus - 1 and j == ContingencyBus - 1:
                    YbusLPost[i, j] -= portion_to_remove
                    YbusLPost[i, to_bus - 1] = 0
                    YbusLPost[to_bus - 1, j] = 0
                    YbusLPost[to_bus - 1, to_bus - 1] -= portion_to_remove

        # Build complete matrix post-fault

        Y1Post = np.hstack((Y, menosYPost))
        Y2Post = np.hstack((menosYPost.T, YbusLPost))
        YcompletePost = np.vstack((Y1Post, Y2Post))

        # Build reduced matrix post-fault

        YbusLinvPost = np.linalg.inv(YbusLPost)  # Matrix D

        YredPost = Y - np.dot(np.dot(menosYPost, YbusLinvPost), menosYPost.T)

        # ------------------------------------------------------------------------------------------------------------------
        #                              FINDING CURRENT VECTOR AND INTERNAL VOLTAGES OF GENERATORS
        # ------------------------------------------------------------------------------------------------------------------

        # Convert voltage and angle to rectangular form

        V_Rect = np.ones(len(V), dtype=complex)  # Complex voltage vector in rectangular form,
        # Initialize an array of ones as complex numbers.

        I = np.zeros(nb, dtype=complex)  # Current injection at system buses, initialize zeros
        Eg = np.zeros(len(Y), dtype=complex)  # Internal voltages of generators
        xlinhad = []  # Transient reactances of generators
        Igen = []  # Current injection at generator buses
        Vgen_bus = []  # Bus voltage at generator buses

        for i in range(len(theta)):  # Convert bus voltages from polar form to rectangular form
            V_Rect[i] = V[i] * np.cos(theta[i]) + 1j * V[i] * np.sin(
                theta[i])  # V * exp(jθ) is a polar form representation

        # YbusLPre_IVg is node admittance matrix takes into account the load equivalent admittance
        I = YbusLPre_IVg @ V_Rect  # @ performs matrix multiplication, calculate Bus Current Injections

        for i in range(nb):
            if BusData['hasGen'][i] != 0:
                Igen.append(complex(I[i]))
                Vgen_bus.append(complex(V_Rect[i]))
                xlinhad.append(1j * (1 / BusData['TransientSusceptance'][i]))

        Igen = np.array(Igen)
        print(abs(Igen))
        Vgen_bus = np.array(Vgen_bus)
        xlinhad = np.array(xlinhad)

        for i in range(len(Igen)):
            Eg[i] = Vgen_bus[i] + Igen[i] * xlinhad[i]  # Eg: internal transient voltage of generator

        print(abs(Eg))

        # ------------------------------------------------------------------------------------------------------------------
        #                              FINDING MATRICES C AND D DURING THE FAULT
        # ------------------------------------------------------------------------------------------------------------------

        # Yred = Y - (menosY)×(YbusL.inverse)×(menosY.T), Igen = Yred×E'
        # Yred is the "effective mutual admittance" between generators only, after eliminating all the load buses.
        # C AND D represent Generator Electrical Coupling Terms, coupling exists only when i≠j.
        CDfault = np.zeros((len(Y), len(Y)), dtype=complex)

        for i in range(len(Y)):
            for j in range(len(Y)):
                if i != j:
                    CDfault[i][j] = abs(Eg[i]) * abs(Eg[j]) * YredFault[i][j]

        Cfault = np.imag(CDfault)
        Dfault = np.real(CDfault)

        # ------------------------------------------------------------------------------------------------------------------
        #                              FINDING MATRICES C AND D POST-FAULT
        # ------------------------------------------------------------------------------------------------------------------

        # YredPost
        CDpost = np.zeros((len(Y), len(Y)), dtype=complex)

        for i in range(len(Y)):
            for j in range(len(Y)):
                if i != j:
                    CDpost[i][j] = abs(Eg[i]) * abs(Eg[j]) * YredPost[i][j]

        Cpost = np.imag(CDpost)
        Dpost = np.real(CDpost)

        # ------------------------------------------------------------------------------------------------------------------
        #                              STORING MECHANICAL POWER IN A VECTOR
        # ------------------------------------------------------------------------------------------------------------------

        Pm = []  # Mechanical Power, generator only

        for i in range(nb):
            if BusData['hasGen'][i] != 0:
                if BusData['hasGen'][i] == 2:
                    Pm.append(0)
                else:
                    Pm.append(Pk[i] / Sbase)
        print(Pm)

        # ------------------------------------------------------------------------------------------------------------------
        #                              STORING INERTIA CONSTANT AND DAMPING IN VECTORS
        # ------------------------------------------------------------------------------------------------------------------

        Damping = []
        M = []
        f = 60

        for i in range(nb):
            if BusData['hasGen'][i] != 0:
                Damping.append(BusData['Damping'][i])
                M.append(BusData['InertiaConst'][i])

        Damping = np.array(Damping)
        M = np.array(M)

        # ------------------------------------------------------------------------------------------------------------------
        #                              DEFINING THE DIFFERENTIAL EQUATIONS FUNCTION
        # ------------------------------------------------------------------------------------------------------------------

        w = np.zeros(ng, dtype=float)
        thetaG = np.angle(Eg)
        # t = 1
        x0 = np.hstack((w, np.angle(Eg)))  # Initial state at fault time

        gen_type = []  # 1 = synchronous, 2 = DFIG (same as 'hasGen' flag)

        for i in range(nb):
            if BusData['hasGen'][i] != 0:
                gen_type.append(BusData['hasGen'][i])

        def differential_pre():
            dw = np.zeros(ng, dtype=float)
            dDelta = np.zeros(ng, dtype=float)

            x = np.hstack((dw, dDelta))
            return x

        def differential_eq(x, Pm, Yred, C, D, Eg, M, Damping):
            dw_dt = np.ones(ng, dtype=float)
            dDelta_dt = np.ones(ng, dtype=float)
            G = np.real(Yred)
            w = x[0:ng]
            delta = x[ng:len(x)]
            # print((w))
            # print((delta))
            # for i in range(ng):
            #     summation = []
            #     for j in range(ng):
            #         if i != j:
            #             summation.append(C[i][j] * np.sin(delta[i] - delta[j]) +
            #                              D[i][j] * np.cos(delta[i] - delta[j]))
            #     dw_dt[i] = (Pm[i] - (abs(Eg[i]) ** 2) * G[i][i] - sum(summation) - Damping[i] * w[i]) / M[i]
            #     # dw/dt = (Pm - Pe - Pi -Dω) / M. Pi = Σ|Ei||Ej|Yred(ij), i≠j.
            #     dDelta_dt[i] = w[i]
            #     # dδ/dt = ω

            for i in range(ng):
                if gen_type[i] != 35:  # Synchronous generator
                    summation = []
                    for j in range(ng):
                        if i != j:
                            summation.append(C[i][j] * np.sin(delta[i] - delta[j]) +
                                             D[i][j] * np.cos(delta[i] - delta[j]))
                    dw_dt[i] = (Pm[i] - (abs(Eg[i]) ** 2) * G[i][i] - sum(summation) - Damping[i] * w[i]) / M[i]

                else:  # DFIG
                    # For DFIG, only consider power injection calculated externally
                    Ps = (abs(Eg[i]) ** 2) * G[i][i]  # Stator electrical power output
                    dw_dt[i] = (Pm[i] - Ps) / (np.pi * f / M[i])

                dDelta_dt[i] = w[i]  # Always update delta (for consistency)

            x = np.hstack((dw_dt, dDelta_dt))
            return x

        def differential_after(x, Pm, Yred, C, D, Eg, M, Damping):
            dw_dt = np.ones(ng, dtype=float)
            dDelta_dt = np.ones(ng, dtype=float)
            G = np.real(Yred)
            w = x[0:ng]
            delta = x[ng:len(x)]
            for i in range(ng):
                if gen_type[i] != 3:  # Synchronous generator
                    summation = []
                    for j in range(ng):
                        if i != j:
                            summation.append(C[i][j] * np.sin(delta[i] - delta[j]) +
                                             D[i][j] * np.cos(delta[i] - delta[j]))
                    dw_dt[i] = (Pm[i] - (abs(Eg[i]) ** 2) * G[i][i] - sum(summation) - Damping[i] * w[i]) / M[i]
                    dDelta_dt[i] = w[i]  # Always update delta (for consistency)


                else:  # DFIG
                    # For DFIG, only consider power injection calculated externally
                    Ps = (abs(Eg[i]) ** 2) * G[i][i]  # Stator electrical power output
                    # dw_dt[i] = (Pm[i] - Ps) / (np.pi * f / M[i])
                    dw_dt[i] = (Pm[i] - Ps - Damping[i] * w[i]) / M[i]  # ωr≠ωs
                    dDelta_dt[i] = 0  # Always update delta (for consistency)

            x = np.hstack((dw_dt, dDelta_dt))
            return x

        # X = differential_eq(t, x0, Pm, YredFault, Cfault, Dfault, Eg, M, Damping)
        # print((X))
        # ------------------------------------------------------------------------------------------------------------------
        #                  SOLVING DIFFERENTIAL EQUATIONS FOR PRE-FAULT PERIOD
        # ------------------------------------------------------------------------------------------------------------------
        fun0 = lambda t, x: differential_pre()
        # fun0 must have two parameters and a return value, the return value should equal to dy/dt
        # fun0 return the return value after :, that is, the return value of differential_pre()

        # solve_ivp:
        # The calling signature is fun(t, y), where t is a scalar and y is an ndarray with len(y) = len(y0).
        # fun must return an array of the same shape as y
        xPRE = solve_ivp(fun=fun0, t_span=[tpre0, tfout0], y0=x0, max_step=maxstep)

        p0 = len(xPRE.y[0])

        # ------------------------------------------------------------------------------------------------------------------
        #                  SOLVING DIFFERENTIAL EQUATIONS FOR THE FAULT PERIOD
        # ------------------------------------------------------------------------------------------------------------------
        fun = lambda t, x: differential_eq(x, Pm, YredFault, Cfault, Dfault, Eg, M, Damping)

        xFAULT = solve_ivp(fun=fun, t_span=[tfout0, tfoult], y0=x0, max_step=maxstep)

        p = len(xFAULT.y[0])
        x1 = xFAULT.y[:, p - 1]  # Initial state at the beginning of post-fault period, last item is p-1

        # ------------------------------------------------------------------------------------------------------------------
        #                  SOLVING DIFFERENTIAL EQUATIONS FOR THE POST-FAULT PERIOD
        # ------------------------------------------------------------------------------------------------------------------

        # fun1 = lambda t, x: differential_eq(t, x, Pm, YredPost, Cpost, Dpost, Eg, M)
        fun1 = lambda t, x: differential_after(x, Pm, YredPost, Cpost, Dpost, Eg, M, Damping)

        xPOST = solve_ivp(fun=fun1, t_span=[tfoult, tAfter], y0=x1, max_step=maxstep)

        p1 = len(xPOST.y[0])

        # ------------------------------------------------------------------------------------------------------------------
        #                                      PLOTTING THE GRAPHS
        # ------------------------------------------------------------------------------------------------------------------

        # The output xPRE.y is the integral of dy/dt over time
        # The i-th row stores all the values of the integral yt[i] of dy[i] over the time period t_span
        # Each column represents all the yt at the current time point, with the same shape as y0(x0)
        Wpre = xPRE.y[0:ng, 0:len(xPRE.y[0])]  # Angular velocity during fault
        Dpre = xPRE.y[ng:2 * ng, 0:len(xPRE.y[0])]  # Delta during fault
        tpre = xPRE.t  # Time during fault

        Wf = xFAULT.y[0:ng, 0:len(xFAULT.y[0])]  # Angular velocity during fault
        Df = xFAULT.y[ng:2 * ng, 0:len(xFAULT.y[0])]  # Delta during fault
        tf = xFAULT.t  # Time during fault

        Wp = xPOST.y[0:ng, 0:len(xPOST.y[0])]  # Angular velocity post-fault
        Dp = xPOST.y[ng:2 * ng, 0:len(xPOST.y[0])]  # Delta post-fault
        tp = xPOST.t  # Time post-fault

        # for i in range(ng):
        #     plt.figure(1)
        #     plt.plot(np.hstack((tpre, tf, tp)), np.hstack((Wpre[i], Wf[i], Wp[i])), label=f'Generator {i + 1}')
        #     plt.title('Angular Velocity vs Time')
        #     plt.xlabel('Time (s)')
        #     plt.ylabel('Angular Velocity (rad/s)')
        #     plt.legend()
        #     plt.grid(True)
        #
        #     plt.figure(2)
        #     plt.plot(np.hstack((tpre, tf, tp)), np.hstack((Dpre[i], Df[i], Dp[i])), label=f'Generator {i + 1}')
        #     plt.title('Generator Internal Angles vs Time')
        #     plt.xlabel('Time (s)')
        #     plt.ylabel('Angle (rad)')
        #     plt.legend()
        #     plt.grid(True)
        #
        # # plt.show()
        # return plt

        time = np.hstack((tpre, tf, tp))

        # --- Figure 1: Angular Velocity vs Time ---
        fig1 = Figure(figsize=(6, 4))
        ax1 = fig1.add_subplot(111)
        for i in range(ng):
            ax1.plot(time, np.hstack((Wpre[i], Wf[i], Wp[i])), label=f'Generator {i + 1}')
        ax1.set_title('Angular Velocity vs Time')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Angular Velocity (rad/s)')
        ax1.legend()
        ax1.grid(True)
        ax1.set_xlim(time[0], time[-1])  # Ensure full x-range

        # --- Figure 2: Internal Angles vs Time ---
        fig2 = Figure(figsize=(6, 4))
        ax2 = fig2.add_subplot(111)
        for i in range(ng):
            ax2.plot(time, np.hstack((Dpre[i], Df[i], Dp[i])), label=f'Generator {i + 1}')
        ax2.set_title('Generator Internal Angles vs Time')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angle (rad)')
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlim(time[0], time[-1])  # Ensure full x-range

        return fig1, fig2

    @staticmethod
    def pebs3(Sbase, Ubase, f, BusData, LineData, CBus, Line, frombus, tobus, fstart, fend, Simtime, mstep):
        Sbase = 100
        Ubase = 138
        Zbase = (Ubase ** 2) / Sbase
        # f = 60

        # Adjust bus numbers for Python indexing (start from 0)
        for k in range(len(LineData['from'])):
            LineData['from'][k] -= 1
        for k in range(len(LineData['to'])):
            LineData['to'][k] -= 1

        # ---------------------------------------------------------------------
        # Extract system size
        nb = len(BusData['num'])  # Number of buses
        nl = len(LineData['from'])  # Number of lines
        npv = 0
        npq = 0
        for i in range(nb):
            if BusData['type'][i] == 1:
                npv += 1
            elif BusData['type'][i] == 0:
                npq += 1

        for i in range(nl):  # Transformer taps 1 means voltage unchanged
            if LineData['tap'][i] == 0:
                LineData['tap'][i] = 1

        # ---------------------------------------------------------------------
        # Normalize and compute net power for each bus

        temp3 = []
        temp4 = []
        bus = {'P': None, 'Q': None, 'bsh': None}
        for i in range(nb):
            temp3.append((BusData['Pgen'][i] - BusData['Pload'][i]) / Sbase)  # calculate pu value
            temp4.append((BusData['Qgen'][i] - BusData['Qload'][i]) / Sbase)  # calculate pu value
            bus['P'] = temp3[:]
            bus['Q'] = temp4[:]
        del temp3
        del temp4

        temp5 = []
        for i in range(nb):
            temp5.append(BusData['Qshunt'][i])
            bus['bsh'] = temp5[:]
        del temp5

        # those two lines can save BusData
        # df = pd.DataFrame(BusData)
        # df.to_excel('BusData.xlsx', index=False)

        # ---------------------------------------------------------------------
        # Convert line parameters to per-unit (pu) system

        for i in range(nl):  # Transformer taps 1 means voltage unchanged
            if LineData['tap'][i] == 0:
                LineData['tap'][i] = 1

        num_tap_normal = 0
        num_phase_shift = 0
        temp3 = []
        for i in range(nl):
            LineData['R'][i] = LineData['R'][i] / Zbase  # ohm to pu
            LineData['Xl'][i] = (LineData['Xl'][i] * 2 * np.pi * f) / Zbase
            LineData['Xc'][i] = (LineData['Xc'][i] * 2 * np.pi * f * Zbase) / 2
            temp3.append(LineData['Xc'][i])  # Calculate Lsh
            LineData['phi'][i] = LineData['phi'][i] * (np.pi / 180)  # Phase-shifter angle to radians
            if LineData['tap'][i] != 1.0:
                num_tap_normal += 1
            elif LineData['phi'][i] != 0.0:
                num_phase_shift += 1

        # ---------------------------------------------------------------------
        # Build Y, G, B matrices for power flow analysis

        Ybus = np.zeros((nb, nb), dtype=complex)
        branch = {'g': None, 'b': None, 'y': None, 'bsh': None}
        temp = []
        temp1 = []
        temp2 = []

        for j in range(nl):
            temp.append(LineData['R'][j] / (np.power(LineData['R'][j], 2) + np.power(LineData['Xl'][j], 2)))
            temp1.append(-LineData['Xl'][j] / (np.power(LineData['R'][j], 2) + np.power(LineData['Xl'][j], 2)))
            temp2.append(temp[j] + temp1[j] * 1j)  # Y = G + jB
            branch['g'] = temp[:]  # real part: conductance
            branch['b'] = temp1[:]  # imaginary part: susceptance
            branch['y'] = temp2[:]  # complex admittance
            branch['bsh'] = temp3[:]
        del temp
        del temp1
        del temp2
        del temp3

        for j in range(nl):
            k = int(LineData['from'][j])
            m = int(LineData['to'][j])
            Ybus[k][m] = -(1 / LineData['tap'][j]) * np.exp(-LineData['phi'][j] * 1j) * branch['y'][j]
            Ybus[m][k] = -(1 / LineData['tap'][j]) * np.exp(LineData['phi'][j] * 1j) * branch['y'][j]
            Ybus[k][k] = Ybus[k][k] + ((1 / LineData['tap'][j]) ** 2) * branch['y'][j] + 1j * branch['bsh'][j]
            Ybus[m][m] = Ybus[m][m] + branch['y'][j] + 1j * branch['bsh'][j]
            # KVL is used for derivation
            # The impedance of the low voltage side is converted to the high voltage side
            # Tap and transformer ratio have the same meaning ,like 0.978:1

        for i in range(nb):
            Ybus[i][i] += bus['bsh'][i] * 1j  # Double-check if this needs to be subtracted instead

        G = np.real(Ybus)
        B = np.imag(Ybus)

        V = np.array(BusData['voltage'], dtype=float).reshape(-1, 1)
        theta = np.array(BusData['angle_radian'], dtype=float).reshape(-1, 1)
        Pk = np.array(BusData['Pk'], dtype=float).reshape(-1, 1)

        # #########################################################################################################################
        #                                         BEGINNING OF PROGRAM
        # #########################################################################################################################
        ContingencyBus = CBus  # Indicates the bus where the contingency occurs
        line = Line  # Defines which line will be removed after the contingency
        from_bus = frombus
        to_bus = tobus

        # Variables for numerical integration

        tpre0 = 0.0
        tfout0 = fstart  # Initial fault time
        tfoult = fend  # Fault clearing time (in seconds)
        tAfter = Simtime  # Maximum post-fault simulation time
        maxstep = mstep  # Integration time step

        # --------------------------------------------------------------------------------------------------------------------------
        #                                   STEP: PRE-FAULT REDUCED MATRIX
        # --------------------------------------------------------------------------------------------------------------------------

        # Building the Yl Matrix

        ng = 0
        TransientAdmittance = []

        for i in range(len(BusData['TransientSusceptance'])):  # Counting number of generators
            if BusData['hasGen'][i] != 0:
                TransientAdmittance.append(-1j * BusData['TransientSusceptance'][i])
                ng += 1

        Y = np.zeros((ng, ng), dtype=complex)  # Diagonal matrix of generator transient admittances (Matrix A)
        TransientAdmittance = np.array(TransientAdmittance)
        np.fill_diagonal(Y, TransientAdmittance)

        print(Y)
        print('=-' * 60)
        print(type(TransientAdmittance))

        S_conj = np.zeros(nb, dtype=complex)  # Conjugate of apparent power
        for i in range(len(BusData['num'])):
            S_conj[i] = BusData['Pload'][i] + 1j * BusData['Qload'][i]

        S_conj = np.conjugate(S_conj)
        Y_const = np.zeros(len(BusData['num']), dtype=complex)  # Load as constant admittance

        for i in range(len(Y_const)):
            Y_const[i] = S_conj[i] / V[i] ** 2
            print(Y_const[i])

        YbusLPre = np.copy(Ybus)

        for i in range(len(BusData['num'])):
            for j in range(len(BusData['num'])):
                if i == j:
                    # print('entered')
                    YbusLPre[i][j] += Y_const[i]

        YbusLPre_IVg = np.copy(YbusLPre)
        gen_index = 0  # Generator index

        for i in range(len(BusData['num'])):
            if BusData['hasGen'][i] != 0:
                gen_index = i
                YbusLPre[gen_index, gen_index] += (-1j * BusData['TransientSusceptance'][i])

        menosY = np.zeros((len(Y), len(BusData['num'])), dtype=complex)  # Matrix B (transpose becomes matrix C)

        generator_buses = []

        for i in range(len(BusData['num'])):
            if BusData['hasGen'][i] != 0:
                generator_buses.append(BusData['num'][i] - 1)

        for i in range(len(Y)):
            for j in range(len(BusData['num'])):
                if j == generator_buses[i]:
                    menosY[i, j] = 1j * BusData['TransientSusceptance'][j]

        # Construct full pre-fault matrix

        Y1Pre = np.hstack((Y, menosY))
        Y2Pre = np.hstack((menosY.T, YbusLPre))
        YcompletePre = np.vstack((Y1Pre, Y2Pre))

        # Construct reduced matrix for pre-fault condition

        YbusLinvPre = np.linalg.inv(YbusLPre)  # Matrix D
        YredPre = Y - np.dot(np.dot(menosY, YbusLinvPre), menosY.T)  # Reduced matrix pre-fault

        # -------------------------------------------------------------------------------------------------------------------
        #                              STEP: REDUCED MATRIX DURING THE FAULT
        # -------------------------------------------------------------------------------------------------------------------

        # Calculate reduced matrix Yred during the fault for a given contingency

        YbusLFault = np.copy(YbusLPre)
        menosYFault = np.copy(menosY)

        for i in range(len(BusData['num'])):
            for j in range(len(BusData['num'])):
                YbusLFault[i][ContingencyBus - 1] = 0
                YbusLFault[ContingencyBus - 1][j] = 0

        for i in range(len(BusData['num'])):
            for j in range(len(BusData['num'])):
                if i == ContingencyBus - 1 and j == ContingencyBus - 1:
                    YbusLFault[i][j] = 1

        flag = 0
        for i in range(ng):
            for j in range(len(BusData['num'])):
                if j == ContingencyBus - 1:
                    if menosYFault[i][j] != 0:
                        menosYFault[i][j] = 0
                        flag = 1
                    if flag == 1:
                        break
            if flag == 1:
                break

        # Build full matrix during fault

        Y1Fault = np.hstack((Y, menosYFault))
        Y2Fault = np.hstack((menosYFault.T, YbusLFault))
        YcompleteFault = np.vstack((Y1Fault, Y2Fault))

        # Build reduced matrix during fault

        YbusLinvFault = np.linalg.inv(YbusLFault)  # Matrix D

        YredFault = Y - np.dot(np.dot(menosYFault, YbusLinvFault), menosYFault.T)

        # ------------------------------------------------------------------------------------------------------------------
        #                              STEP: REDUCED MATRIX POST-FAULT
        # ------------------------------------------------------------------------------------------------------------------

        YbusLPost = np.copy(YbusLPre)
        menosYPost = np.copy(menosY)

        # Remove the branch corresponding to the contingency

        portion_to_remove = complex(1 + 1j)
        for i in range(nl):
            if i == line - 1:
                portion_to_remove = branch['y'][i]  # Store admittance to remove from Ybus
        print(portion_to_remove)

        for i in range(nb):
            for j in range(nb):
                if i == ContingencyBus - 1 and j == ContingencyBus - 1:
                    YbusLPost[i, j] -= portion_to_remove
                    YbusLPost[i, to_bus - 1] = 0
                    YbusLPost[to_bus - 1, j] = 0
                    YbusLPost[to_bus - 1, to_bus - 1] -= portion_to_remove

        # Build full matrix post-fault

        Y1Post = np.hstack((Y, menosYPost))
        Y2Post = np.hstack((menosYPost.T, YbusLPost))
        YcompletePost = np.vstack((Y1Post, Y2Post))

        # Build reduced matrix post-fault

        YbusLinvPost = np.linalg.inv(YbusLPost)  # Matrix D

        YredPost = Y - np.dot(np.dot(menosYPost, YbusLinvPost), menosYPost.T)

        # ------------------------------------------------------------------------------------------------------------------
        #                              FIND GENERATOR CURRENT INJECTION AND INTERNAL VOLTAGES
        # ------------------------------------------------------------------------------------------------------------------

        # Convert voltages and angles to rectangular form

        V_Rect = np.ones(len(V), dtype=complex)  # Complex voltage vector in rectangular form
        I = np.zeros(nb, dtype=complex)  # Current injection at system buses
        Eg = np.zeros(len(Y), dtype=complex)  # Internal generator voltages
        xlinhad = []  # Generator transient reactances
        Igen = []  # Generator current injections
        Vgen_bus = []  # Generator bus voltages

        for i in range(len(theta)):
            V_Rect[i] = V[i] * np.cos(theta[i]) + 1j * V[i] * np.sin(
                theta[i])  # V*exp(jθ) is a polar form representation

        I = YbusLPre_IVg @ V_Rect  # @ performs matrix multiplication

        for i in range(nb):
            if BusData['hasGen'][i] != 0:
                Igen.append(complex(I[i]))
                Vgen_bus.append(complex(V_Rect[i]))
                xlinhad.append(1j * (1 / BusData['TransientSusceptance'][i]))

        Igen = np.array(Igen)
        print(abs(Igen))
        Vgen_bus = np.array(Vgen_bus)
        xlinhad = np.array(xlinhad)

        for i in range(len(Igen)):
            Eg[i] = Vgen_bus[i] + Igen[i] * xlinhad[i]

        print(abs(Eg))

        # ------------------------------------------------------------------------------------------------------------------
        #                              FINDING MATRICES C AND D DURING THE FAULT
        # ------------------------------------------------------------------------------------------------------------------

        CDfault = np.zeros((len(Y), len(Y)), dtype=complex)

        for i in range(len(Y)):
            for j in range(len(Y)):
                if i != j:
                    CDfault[i][j] = abs(Eg[i]) * abs(Eg[j]) * YredFault[i][j]

        Cfault = np.imag(CDfault)
        Dfault = np.real(CDfault)

        # ------------------------------------------------------------------------------------------------------------------
        #                              FINDING MATRICES C AND D POST-FAULT
        # ------------------------------------------------------------------------------------------------------------------

        CDpost = np.zeros((len(Y), len(Y)), dtype=complex)

        for i in range(len(Y)):
            for j in range(len(Y)):
                if i != j:
                    CDpost[i][j] = abs(Eg[i]) * abs(Eg[j]) * YredPost[i][j]

        Cpost = np.imag(CDpost)
        Dpost = np.real(CDpost)

        # ------------------------------------------------------------------------------------------------------------------
        #                              STORING MECHANICAL POWER IN A VECTOR
        # ------------------------------------------------------------------------------------------------------------------

        Pm = []  # Mechanical Power

        for i in range(nb):
            if BusData['hasGen'][i] != 0:
                if BusData['hasGen'][i] == 2:
                    Pm.append(0)
                else:
                    Pm.append(Pk[i] / Sbase)

        print(Pm)

        # ------------------------------------------------------------------------------------------------------------------
        #                              STORING INERTIA CONSTANT AND DAMPING IN VECTORS
        # ------------------------------------------------------------------------------------------------------------------

        Damping = []
        M = []

        for i in range(nb):
            if BusData['hasGen'][i] != 0:
                Damping.append(BusData['Damping'][i])
                M.append(BusData['InertiaConst'][i])

        Damping = np.array(Damping)
        M = np.array(M)

        # ------------------------------------------------------------------------------------------------------------------
        #                              DEFINING THE DIFFERENTIAL EQUATIONS FUNCTION
        # ------------------------------------------------------------------------------------------------------------------

        w = np.zeros(ng, dtype=float)
        thetaG = np.angle(Eg)
        t = 1
        x0 = np.hstack((w, np.angle(Eg)))  # Initial state at the fault moment

        def differential_eq(t, x, Pm, Yred, C, D, Eg, M):
            dw = np.ones(ng, dtype=float)
            dDelta = np.ones(ng, dtype=float)
            G = np.real(Yred)
            w = x[0:ng]
            delta = x[ng:len(x)]
            # print((w))
            # print((delta))
            for i in range(ng):
                summation = []
                for j in range(ng):
                    if i != j:
                        summation.append(C[i][j] * np.sin(delta[i] - delta[j]) + D[i][j] * np.cos(delta[i] - delta[j]))
                dw[i] = (Pm[i] - (abs(Eg[i]) ** 2) * G[i][i] - sum(summation)) / M[i]
                dDelta[i] = w[i]

            x = np.hstack((dw, dDelta))
            return x

        # X = differential_eq(t, x0, Pm, YredFault, Cfault, Dfault, Eg, M)
        # print((X))

        # ------------------------------------------------------------------------------------------------------------------
        #                  SOLVING DIFFERENTIAL EQUATIONS FOR THE FAULT PERIOD
        # ------------------------------------------------------------------------------------------------------------------

        fun = lambda t, x: differential_eq(t, x, Pm, YredFault, Cfault, Dfault, Eg, M)

        xFAULT = solve_ivp(fun=fun, t_span=[tfout0, tfoult], y0=x0, max_step=maxstep)

        p = len(xFAULT.y[0])
        x1 = xFAULT.y[:, p - 1]  # Initial state at the beginning of the post-fault period, last item is p-1

        Wf = xFAULT.y[0:ng, 0:len(xFAULT.y[0])]  # Omega during fault
        Df = xFAULT.y[ng:2 * ng, 0:len(xFAULT.y[0])]  # Delta during fault
        tf = xFAULT.t  # Fault time vector

        # ------------------------------------------------------------------------------------------------------------------
        #                  CALCULATING COA REFERENCES FOR THE FAULT PERIOD
        # ------------------------------------------------------------------------------------------------------------------

        print(Wf[:, 0])
        print(Df[:, 0])
        print(len(Wf[0, :]))

        # Center of Angle (COA) for the fault period:

        DeltaCOAF = []  # center-of-angle rotor angle at time t[i]
        DeltaOmegaCOAF = []  # center-of-angle angular velocity at time t[i]
        DeltaRefF = np.zeros((ng, len(Df[0, :])), dtype=float)  # rotor angle under COA frame
        OmegaRefF = np.zeros((ng, len(Wf[0, :])), dtype=float)  # angular velocity under COA frame

        MT = sum(M)  # total system inertia
        sum1 = 0
        sum2 = 0

        for i in range(len(Wf[0, :])):
            for j in range(ng):
                sum1 += M[j] * Wf[j, i]  # M-weighted ω_j
                sum2 += M[j] * Df[j, i]  # M-weighted δ_j

            # δ_COA(ave) = (ΣM[j]×δ[j])/MT  [ti]
            # ω_COA(ave) = (ΣM[j]×ω[j])/MT  [ti]
            DeltaCOAF.append((1 / MT) * sum2)  # weighted average δ_COA(ave)
            DeltaOmegaCOAF.append((1 / MT) * sum1)  # weighted average θ_COA(ave)

            # Transform each generator’s values to COA frame  [ti]
            # δ_reff(COA)[k] = δ[k] - δ_COA(ave)  [ti]
            # ω_reff(COA)[k] = ω[k] - ω_COA(ave)  [ti]
            for k in range(ng):  # Apply COA shift to each generator
                DeltaRefF[k, i] = Df[k, i] - DeltaCOAF[i]  # delta shifted to COA
                OmegaRefF[k, i] = Wf[k, i] - DeltaOmegaCOAF[i]  # omega shifted to COA

            sum1 = 0
            sum2 = 0

        # ------------------------------------------------------------------------------------------------------------------
        #                         CALCULATING KINETIC AND POTENTIAL ENERGY
        # ------------------------------------------------------------------------------------------------------------------

        thetaG_COA = []

        # Shifts all generator rotor angles relative to the COA at the start of the fault.
        # θ_initial[i](COA) = θ_initial[i] - δ_COA(ave)[0]
        for i in range(ng):
            thetaG_COA.append(thetaG[i] - DeltaCOAF[0])
            # thetaG[i]-initial rotor angle of generator i at the beginning of the fault
            # DeltaCOAF[0]-COA rotor angle at the start of fault period
        sum3 = 0

        Ec = []  # total kinetic energy of all generators at one time step
        Ep = []  # Potential energy

        templen0 = len(Wf[0, :])  # number of time points

        # Ec(sum3) = Σ0.5×M[j]×(ω_ref[j])^2  [ti]
        for i in range(len(Wf[0, :])):
            for j in range(ng):
                sum3 += 0.5 * M[j] * OmegaRefF[j, i] ** 2
                # OmegaRefF-angular velocity under COA frame
                # ω_reff(COA)[k] = ω[k] - ω_COA(ave)  [ti]
            Ec.append(sum3)  # Ec.append(total kinetic energy of all generators at one time step)
            sum3 = 0

        # print(Ec)
        del templen0
        sum4 = 0
        sum4_components = []

        # Potential energy from own rotor displacement and mechanical imbalance
        # At each time step, potential energy from each generator itself:
        # sum4 = Σ(Pm-P_dissipated)×(δ_reff(COA)-θ_initial(COA))  [ti]
        for i in range(len(Wf[0, :])):
            for j in range(ng):
                sum4 += ((Pm[j] - (abs(Eg[j]) ** 2) * np.real(YredPost[j, j])) *
                         (DeltaRefF[j, i] - thetaG_COA[j]))
                # Pm[j] mechanical power input of generator j
                # |Eg[j]| magnitude of generator j’s internal voltage
                # Re(YredPost[j, j]) reduced post-fault Y conductance
                # δ_reff[j](COA) = δ[j] - δ_COA(ave)  [ti]
                # θ_initial[j](COA) = θ_initial[j] - δ_COA(ave)   [t0]
            sum4_components.append(sum4)
            sum4 = 0

        print(sum4_components[0])
        sum5 = 0
        sum5_temp = 0
        sum5_components = []
        print(sum4)

        # Mutual potential energy between generators due to rotor angle differences
        # sum5 = Σ(0, ng-1)Σ(i+1, ng)Cpost[i, j]×(cos(δ_reff[i](COA)-δ_reff[j](COA)) - cos(θ_init[i](COA)-θ_init[j](COA)))
        for k in range(len(Wf[0, :])):  # For each time step k
            for i in range(ng - 1):  # For each generator i
                for j in range(i + 1, ng):  # For each other generator j (i < j --> avoid repeat and self-term)
                    sum5_temp += Cpost[i, j] * (np.cos(DeltaRefF[i, k] - DeltaRefF[j, k]) -
                                                np.cos(thetaG_COA[i] - thetaG_COA[j]))
                    # Cpost[i, j] = |Eg[i]×Eg[j]|×Im(Yred[i][j])  i≠j
                sum5 += sum5_temp
                sum5_temp = 0
            sum5_components.append(sum5)
            sum5 = 0

        print(sum5_components)

        sum6 = 0
        sum6_temp = 0
        sum6_components = []

        # sum6 = Σ(0,ng-1)Σ(i+1,ng)Dpost[i, j]×(num_ij/den_ij)×(sin(Δδ[ij](COA)) - sin(Δθ[ij](COA)))  t[k]
        for k in range(len(Wf[0, :])):  # Time step k
            for i in range(ng - 1):  # Generator i
                for j in range(i + 1, ng):  # Generator j > i
                    if k == 0:
                        break
                    else:
                        numerator = (DeltaRefF[i, k] + DeltaRefF[j, k] - thetaG_COA[i] - thetaG_COA[j])
                        # numerator: total difference between the current phase angle and the initial angle
                        # the smaller the difference with the initial value, the smaller sum6 is.
                        # numerator = ((δ_reff[i](COA) - θ_init[i](COA)) + (δ_reff[j](COA) - θ_init[j](COA)))  t[k]
                        denominator = (DeltaRefF[i, k] - DeltaRefF[j, k] - thetaG_COA[i] + thetaG_COA[j])
                        # denominator: difference in the phase angle change of different engines.
                        # the smaller den is, the more synchronization deviates from the initial value, and the larger sum6 is.
                        # denominator = ((δ_reff[i](COA) - δ_reff[j](COA)) - (θ_init[i](COA) - θ_init[j](COA)))  t[k]
                        sum6_temp += (Dpost[i, j] * (numerator / denominator) *
                                      (np.sin(DeltaRefF[i, k] - DeltaRefF[j, k]) - np.sin(
                                          thetaG_COA[i] - thetaG_COA[j])))
                        # Dpost[i, j] = |Eg[i]×Eg[j]|×Re(Yred[i][j])  i≠j
                        # (sin(Δδ[ij](COA)) - sin(Δθ[ij](COA)))/den_ij in line with the definition of d(sin(x))/dx,
                        # when the two generators change synchronously, the value of the formula is 1
                sum6 += sum6_temp
                sum6_temp = 0
            sum6_components.append(sum6)
            sum6 = 0

        sum6_components[0] = 0
        print(len(Ec))

        # Calculate Potential Energy

        # Ep = − (sum4 + sum5) + sum6
        # Negative → restoring nature
        # Positive → accumulated risk of instability
        for i in range(len(Ec)):
            Ep.append(-sum4_components[i] - sum5_components[i] + sum6_components[i])

        # print(V)

        # ------------------------------------------------------------------------------------------------------------------
        #                                CALCULATING CRITICAL ENERGY VALUE (Vcritico)
        # ------------------------------------------------------------------------------------------------------------------

        maximum = 20
        higher = 0
        lower = 0
        for i in range(len(Ep)):
            if Ep[i] > 0:
                maximum = Ep[i]
                if i != 0:
                    if Ep[i + 1] > maximum:
                        maximum = Ep[i + 1]
                        higher = Ep[i + 6]
                        lower = Ep[i]
                    elif Ep[i + 1] < maximum:
                        break

        Vcritico = maximum

        # print(Vcritico)
        # print(higher)
        # print(lower)

        # ------------------------------------------------------------------------------------------------------------------
        #                  COMPUTING TOTAL ENERGY AND COMPARING TO CRITICAL ENERGY (Vcritico)
        # ------------------------------------------------------------------------------------------------------------------

        Et = []
        t = 0
        Tcritico = 0

        for i in range(len(Ep)):
            Et.append(Ec[i] + Ep[i])

        for i in range(len(Ep)):
            t = Et[i]
            if Et[i + 1] > Vcritico:
                Tcritico = tf[i]
                break

        print(Tcritico)

        # ------------------------------------------------------------------------------------------------------------------
        #           SOLVING DIFFERENTIAL EQUATIONS FOR THE POST-FAULT PERIOD (AFTER CRITICAL ENERGY TIME)
        # ------------------------------------------------------------------------------------------------------------------

        fun1 = lambda t, x: differential_eq(t, x, Pm, YredPost, Cpost, Dpost, Eg, M)

        xPOST = solve_ivp(fun=fun1, t_span=[Tcritico, tAfter], y0=x1, max_step=maxstep)

        p1 = len(xPOST.y[0])

        # ------------------------------------------------------------------------------------------------------------------
        #                                         PLOTTING THE GRAPHS
        # ------------------------------------------------------------------------------------------------------------------

        Wp = xPOST.y[0:ng, 0:len(xPOST.y[0])]  # Angular velocity post-fault
        Dp = xPOST.y[ng:2 * ng, 0:len(xPOST.y[0])]  # Internal angle post-fault
        tp = xPOST.t  # Time vector post-fault

        # ------------------------------------------------------------------------------------------------------------------
        #           CALCULATING COA REFERENCES FOR THE POST-FAULT PERIOD
        # ------------------------------------------------------------------------------------------------------------------

        dDeltaCOAPF = []
        dDeltaOmegaCOAPF = []
        DeltaRefPF = np.zeros((ng, len(Dp[0, :])), dtype=float)
        OmegaRefPF = np.zeros((ng, len(Wp[0, :])), dtype=float)

        sum1 = 0
        sum2 = 0

        for i in range(len(Wp[0, :])):
            for j in range(ng):
                sum1 += M[j] * Wp[j, i]
                sum2 += M[j] * Dp[j, i]

            dDeltaCOAPF.append((1 / MT) * sum2)  # delta0 for each iteration (post-fault)
            dDeltaOmegaCOAPF.append((1 / MT) * sum1)  # omega0 for each iteration (post-fault)

            for k in range(ng):
                DeltaRefPF[k, i] = Dp[k, i] - dDeltaCOAPF[i]
                OmegaRefPF[k, i] = Wp[k, i] - dDeltaOmegaCOAPF[i]

            sum1 = 0
            sum2 = 0

        # ------------------------------------------------------------------------------------------------------------------
        #           PREPARING TO PLOT ENERGY VS TIME UNTIL CRITICAL ENERGY IS REACHED
        # ------------------------------------------------------------------------------------------------------------------

        pltEt = []
        pltTc = []

        for i in range(len(Et)):
            if Et[i] <= Vcritico:
                pltEt.append(Et[i])
                pltTc.append(tf[i])
            elif Et[i] > Vcritico:
                break

        print(len(Ec))
        print(len(Ep))
        print(len(Et))
        print(len(tf))

        # ------------------------------------------------------------------------------------------------------------------
        #                                         PLOTTING THE GRAPHS
        # ------------------------------------------------------------------------------------------------------------------

        # plt.figure(3)
        # plt.rcParams.update({'font.size': 10})
        # plt.plot(tf, Et, label='Total Energy (Et)')
        # plt.plot(tf, Ec, label='Kinetic Energy (Ec)')
        # plt.plot(tf, Ep, label='Potential Energy (Ep)')
        # plt.title('Energy vs Time')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Energy (p.u.)')
        # plt.legend()
        # plt.grid(True)
        #
        # return plt

        # fig = Figure(figsize=(6, 4))
        # ax = fig.add_subplot(111)
        # ax.plot(tf, Et, label='Total Energy (Et)')
        # ax.plot(tf, Ec, label='Kinetic Energy (Ec)')
        # ax.plot(tf, Ep, label='Potential Energy (Ep)')
        # ax.set_title('Energy vs Time')
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('Energy (p.u.)')
        # ax.legend()
        # ax.grid(True)

        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.plot(tf, Et, label='Total Energy (Et)')
        ax.plot(tf, Ec, label='Kinetic Energy (Ec)')
        ax.plot(tf, Ep, label='Potential Energy (Ep)')
        ax.set_title('Energy vs Time')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy (p.u.)')
        ax.set_xlim([tf[0], tf[-1]])  # Ensures full time range is shown
        ax.legend()
        ax.grid(True)

        return fig

